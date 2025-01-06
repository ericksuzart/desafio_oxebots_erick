import math

import numpy
import rclpy
from desafio_oxebots_erick_interfaces.action import MoveBase
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from tf_transformations import euler_from_quaternion

# Constants and States definition
DISTANCE_THRESHOLD = 1.0
END_POSITION_Y = 10

START = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3
END = 4


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Subscribers
        self.odom_sub = None
        self.laser_sub = None
        self.imu_sub = None

        # Publisher
        self.cmd_vel_pub = None

        # Action client
        self.move_base_client = None
        self.action_complete = True

        # sensor data
        self.front_distance = numpy.inf
        self.left_distance = numpy.inf
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        # robot state
        self.position = (0.0, 0.0)
        self.orientation = 0.0

        self.setup()

        self.get_logger().info('RobotController node initialized.')

    def setup(self):
        """Sets up subscribers, publishers, etc. to configure the node"""

        # subscriber for receiving data from the robot
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile=10
        )
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, qos_profile=10
        )
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, qos_profile=10)

        # publisher to move the robot
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile=10)

        # action server for handling actions to move the robot
        self.move_base_server = ActionServer(
            self,
            MoveBase,
            '/move_base_action',
            execute_callback=self.move_base_execute,
            goal_callback=self.move_base_handle_goal,
            cancel_callback=self.move_base_handle_cancel,
            handle_accepted_callback=self.move_base_handle_accepted,
        )

        # action client for sending action goals to move the robot
        self.move_base_client = ActionClient(self, MoveBase, '/move_base_action')

        # timer that updates the state machine every 0.1 s
        self.timer = self.create_timer(0.1, self.state_machine)

        self.current_state = START
        self.next_state = START

        # rate that determines the frequency of the loops
        self.loop_rate = self.create_rate(100, self.get_clock())

    #
    # Data processing callbacks
    #

    def odom_callback(self, msg: Odometry):
        """Update (x,y) position and yaw orientation from /odom."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        _, _, yaw = euler_from_quaternion(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ]
        )

        self.position = (x, y)
        self.orientation = yaw

    def laser_callback(self, msg: LaserScan):
        """Extract the front and left distances from the laser sensor messages"""
        self.front_distance = msg.ranges[0]
        self.left_distance = msg.ranges[1]

    def imu_callback(self, msg: Imu):
        """Extract linear/angular velocity from IMU."""
        self.linear_velocity = msg.linear_acceleration.x
        self.angular_velocity = msg.angular_velocity.z

    #
    # Helper methods
    #
    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to (-pi, pi)."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def angle_diff(self, a, b):
        """Compute difference between two angles, result in (-pi, +pi)."""
        return self.normalize_angle(a - b)

    def publish_cmd_vel(self, lin_vel, ang_vel):
        """Publish a Twist message to /cmd_vel."""
        twist = Twist()
        twist.linear.x = lin_vel
        twist.angular.z = ang_vel
        self.cmd_vel_pub.publish(twist)

    def stop_robot(self):
        """Stop the robot by publishing zero velocities."""
        self.publish_cmd_vel(0.0, 0.0)

    #
    # Action Server: https://docs.ros.org/en/humble/Tutorials/Intermediate/Writing-an-Action-Server-Client/Py.html
    #
    def move_base_handle_goal(self, goal: MoveBase.Goal) -> GoalResponse:
        """Processes action goal requests"""
        return GoalResponse.ACCEPT

    def move_base_handle_cancel(self, goal: MoveBase.Goal) -> CancelResponse:
        """Processes action cancel requests"""
        return CancelResponse.ACCEPT

    def move_base_handle_accepted(self, goal: MoveBase.Goal):
        """Processes accepted action goal requests"""

        # execute action in a separate thread to avoid blocking
        goal.execute()

    async def move_base_execute(self, goal_handle: MoveBase.Goal) -> MoveBase.Result:
        """
        Asynchronously executes the MoveBase action goal to move or turn a robot.
        """
        # Goal is the requested movement
        goal = goal_handle.request
        movement_type = goal.goal_move

        # Prepare the result and feedback message
        result = MoveBase.Result()
        feedback = MoveBase.Feedback()
        feedback.percentage_completed = 0.0

        # Stop command: no movement, immediately succeed
        if movement_type == goal.STOP:
            self.stop_robot()
            result.success = True
            return result

        # Depending on forward/backward or left/right, we define:
        #   - target_distance (in meters), or
        #   - target_angle (in radians).
        target_distance = 0.0
        target_angle = 0.0

        # Save initial state so we can figure out how far/angle we've moved
        start_x, start_y = self.position
        start_yaw = self.orientation

        if movement_type == goal.MOVE_FORWARD:
            target_distance = 1.0  # 1 meter

        elif movement_type == goal.TURN_LEFT:
            # Snap start_yaw to nearest multiple of pi/2
            snapped_yaw = (math.pi / 2.0) * round(start_yaw / (math.pi / 2.0))
            # Turn left by pi/2
            final_yaw = snapped_yaw + math.pi / 2.0
            # Compute the actual angle difference we must turn
            target_angle = self.angle_diff(final_yaw, start_yaw)

        elif movement_type == goal.TURN_RIGHT:
            # Snap start_yaw to nearest multiple of pi/2
            snapped_yaw = (math.pi / 2.0) * round(start_yaw / (math.pi / 2.0))
            # Turn right by pi/2
            final_yaw = snapped_yaw - math.pi / 2.0
            # Compute the actual angle difference we must turn
            target_angle = self.angle_diff(final_yaw, start_yaw)

        # Define the speed for the robot to move
        linear_speed = 0.1  # m/s
        max_angular_speed = 0.5  # rad/s

        # Decide if we are moving straight or turning
        # Use a small loop to move until close to target
        if abs(target_distance) > 0:
            # We are moving forward/backward
            distance_to_travel = abs(target_distance)

            while rclpy.ok():
                # Check if the goal was canceled
                if goal_handle.is_cancel_requested:
                    self.stop_robot()
                    result.success = False
                    goal_handle.canceled()
                    self.get_logger().info('MoveBase canceled (distance).')
                    return result

                current_x, current_y = self.position
                traveled = math.sqrt((current_x - start_x) ** 2 + (current_y - start_y) ** 2)

                # If we've traveled the required distance, break
                if traveled >= distance_to_travel:
                    break

                # If front distance is at or below 0.5 m, stop moving
                if self.front_distance <= 0.5:
                    break

                # Publish velocity commands to move the robot
                self.publish_cmd_vel(linear_speed, 0.0)

                # Provide some feedback for the action
                progress = min(traveled / distance_to_travel, 1.0)
                feedback.percentage_completed = progress * 100.0
                goal_handle.publish_feedback(feedback)

                # Sleep for a short time
                self.loop_rate.sleep()

            # Stop the robot once the distance is reached or we are blocked
            self.stop_robot()

        elif abs(target_angle) > 0:
            # Desired final yaw
            goal_yaw = self.normalize_angle(start_yaw + target_angle)

            # A simple P-controller gain for turning
            Kp = 0.8

            # We will break once error < tolerance
            angle_tolerance = 0.01  # rad (~ 0.57°)

            # Loop until we are within the tolerance
            while rclpy.ok():
                # Check if the goal was canceled
                if goal_handle.is_cancel_requested:
                    self.stop_robot()
                    result.success = False
                    goal_handle.canceled()
                    self.get_logger().info('MoveBase canceled (angle).')
                    return result

                # Current error
                error = self.angle_diff(goal_yaw, self.orientation)

                # Are we close enough?
                if abs(error) < angle_tolerance:
                    break

                # P-controller for angular velocity
                ang_vel = Kp * error

                # Clamp the speed to avoid going too fast
                ang_vel = max(min(ang_vel, max_angular_speed), -max_angular_speed)

                # Publish velocity commands to turn the robot
                self.publish_cmd_vel(0.0, ang_vel)

                # Provide some feedback
                turned_so_far = self.angle_diff(self.orientation, start_yaw)
                progress = (
                    abs(turned_so_far / target_angle) if target_angle != 0 else 0.0
                )  # avoid division by zero
                feedback.percentage_completed = min(progress, 1.0) * 100.0
                goal_handle.publish_feedback(feedback)

                # Sleep for a short time
                self.loop_rate.sleep()

            # Stop the robot once the angle is reached
            self.stop_robot()

        # If we made it this far, we consider the action succeeded
        result.success = True
        goal_handle.succeed()

        return result

    #
    # Action Client: https://docs.ros.org/en/humble/Tutorials/Intermediate/Writing-an-Action-Server-Client/Py.html
    #
    def send_move_goal(self, goal: MoveBase.Goal):
        """Send a goal to the MoveBase action server."""
        self.action_complete = False
        self.move_base_client.wait_for_server()
        self._send_goal_future = self.move_base_client.send_goal_async(
            goal, feedback_callback=self.move_base_feedback_callback
        )
        self._send_goal_future.add_done_callback(self.move_base_goal_response_callback)

    def move_base_goal_response_callback(self, future):
        """Callback for goal response."""
        goal_handle = future.result()

        if not goal_handle.accepted:
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.move_base_get_result_callback)

    def move_base_get_result_callback(self, future):
        """Callback for getting the result."""
        result = future.result().result
        if result.success:
            self.action_complete = True

    def move_base_feedback_callback(self, feedback_msg: MoveBase.Feedback):
        """Callback for feedback."""
        pass

    def state_machine(self):
        """State machine to control the robot based on Hand On Wall algorithm"""

        if not self.action_complete or self.current_state == END:
            return

        # State transitions
        if self.position[1] >= END_POSITION_Y:
            self.next_state = END

        elif self.current_state == START:
            if self.front_distance > DISTANCE_THRESHOLD:
                self.next_state = START
            else:
                self.next_state = TURN_RIGHT

        elif self.current_state == TURN_RIGHT or self.current_state == TURN_LEFT:
            self.next_state = MOVE_FORWARD

        elif self.current_state == MOVE_FORWARD:
            if self.left_distance > DISTANCE_THRESHOLD:
                self.next_state = TURN_LEFT
            elif self.front_distance <= DISTANCE_THRESHOLD:
                self.next_state = TURN_RIGHT
            else:
                self.next_state = MOVE_FORWARD

        elif self.current_state == END:
            self.next_state = END

        # State actions
        if self.next_state == START:
            self.get_logger().info('Starting the maze navigation.')
            self.send_move_goal(MoveBase.Goal(goal_move=MoveBase.Goal.MOVE_FORWARD))
        elif self.next_state == TURN_RIGHT:
            self.get_logger().info('Turning right.')
            self.send_move_goal(MoveBase.Goal(goal_move=MoveBase.Goal.TURN_RIGHT))
        elif self.next_state == TURN_LEFT:
            self.get_logger().info('Turning left.')
            self.send_move_goal(MoveBase.Goal(goal_move=MoveBase.Goal.TURN_LEFT))
        elif self.next_state == MOVE_FORWARD:
            self.get_logger().info('Moving forward.')
            self.send_move_goal(MoveBase.Goal(goal_move=MoveBase.Goal.MOVE_FORWARD))
        elif self.next_state == END:
            self.get_logger().info('Ending the maze navigation.')
            self.send_move_goal(MoveBase.Goal(goal_move=MoveBase.Goal.STOP))

        self.current_state = self.next_state


def main():
    rclpy.init()
    node = RobotController()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

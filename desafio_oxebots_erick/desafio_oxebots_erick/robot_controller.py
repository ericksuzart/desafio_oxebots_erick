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
DISTANCE_THRESHOLD = 1.0 # meters
END_POSITION_Y = 10 # meters
TB3_MAX_LIN_VEL = 0.1 # m/s
TB3_MAX_ANG_VEL = 1.0 # rad/s

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
        Executes robot motion commands based on the goal parameter.

        This function checks the requested motion goal and performs the appropriate action:
        - STOP: Halts the robot’s movement.
        - MOVE_FORWARD: Moves the robot forward a fixed distance.
        - TURN_LEFT or TURN_RIGHT: Rotates the robot in place to the nearest 90° multiple,
            left or right.
        """
        goal = goal_handle.request
        result = MoveBase.Result()
        feedback = MoveBase.Feedback()
        feedback.percentage_completed = 0.0

        if goal.goal_move == goal.STOP:
            self.stop_robot()
            result.success = True
            return result

        if goal.goal_move == goal.MOVE_FORWARD:
            success = await self.move_forward(goal_handle, 1.0, feedback)
        else:
            # TURN_LEFT or TURN_RIGHT
            start_yaw = self.orientation
            snapped_yaw = (math.pi / 2.0) * round(start_yaw / (math.pi / 2.0))
            delta = math.pi / 2.0 if goal.goal_move == goal.TURN_LEFT else -math.pi / 2.0
            final_yaw = snapped_yaw + delta
            target_angle = self.angle_diff(final_yaw, start_yaw)
            success = await self.rotate_in_place(goal_handle, target_angle, feedback)

        result.success = success
        if success:
            goal_handle.succeed()
        else:
            goal_handle.canceled()
        return result

    async def move_forward(self, goal_handle, target_distance, feedback):
        """
        Asynchronously moves the robot forward until the specified distance is traveled or
        an obstacle is detected within a safe range.
        """
        start_x, start_y = self.position
        distance_to_travel = abs(target_distance)
        linear_speed = TB3_MAX_LIN_VEL

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.stop_robot()
                return False

            current_x, current_y = self.position
            traveled = math.sqrt((current_x - start_x) ** 2 + (current_y - start_y) ** 2)

            if traveled >= distance_to_travel or self.front_distance <= 0.5:
                break

            self.publish_cmd_vel(linear_speed, 0.0)
            feedback.percentage_completed = min(traveled / distance_to_travel, 1.0) * 100.0
            goal_handle.publish_feedback(feedback)
            self.loop_rate.sleep()

        self.stop_robot()
        return True

    async def rotate_in_place(self, goal_handle, target_angle, feedback):
        """
        Asynchronously rotates the robot in place until the current yaw angle reaches the given
        target yaw angle within a specified tolerance.
        """
        start_yaw = self.orientation
        goal_yaw = self.normalize_angle(start_yaw + target_angle)
        Kp = 0.8
        angle_tolerance = 0.01
        max_angular_speed = TB3_MAX_ANG_VEL

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.stop_robot()
                return False

            error = self.angle_diff(goal_yaw, self.orientation)

            if abs(error) < angle_tolerance:
                break

            ang_vel = Kp * error
            ang_vel = max(min(ang_vel, max_angular_speed), -max_angular_speed)
            self.publish_cmd_vel(0.0, ang_vel)

            turned = self.angle_diff(self.orientation, start_yaw)
            progress = abs(turned / target_angle) if target_angle else 0.0
            feedback.percentage_completed = min(progress, 1.0) * 100.0
            goal_handle.publish_feedback(feedback)
            self.loop_rate.sleep()

        self.stop_robot()
        return True

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

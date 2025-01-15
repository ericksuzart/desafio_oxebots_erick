import math
from enum import Enum

import numpy
import rclpy
from desafio_oxebots_erick_interfaces.action import MoveBase
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion


class FSM_state(Enum):
    START = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    END = 4


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Declare ROS 2 parameters for configurable constants
        self.declare_parameter('distance_threshold', 1.0)
        self.declare_parameter('end_position_y', 10.0)
        self.declare_parameter('bot_max_lin_vel', 1.0)
        self.declare_parameter('bot_max_ang_vel', 3.0)
        self.declare_parameter('update_frequency', 100)

        # Retrieve parameter values
        self.distance_threshold = self.get_parameter('distance_threshold').value
        self.end_position_y = self.get_parameter('end_position_y').value
        self.bot_max_lin_vel = self.get_parameter('bot_max_lin_vel').value
        self.bot_max_ang_vel = self.get_parameter('bot_max_ang_vel').value
        self.update_frequency = self.get_parameter('update_frequency').value

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

        # robot state
        self.position = (0.0, 0.0)
        self.orientation = 0.0
        self.current_speed = 0.0

        # state machine

        self.STATE_ACTION = {
            FSM_state.START: ('Starting the maze navigation.', MoveBase.Goal.MOVE_FORWARD),
            FSM_state.TURN_RIGHT: ('Turning right.', MoveBase.Goal.TURN_RIGHT),
            FSM_state.TURN_LEFT: ('Turning left.', MoveBase.Goal.TURN_LEFT),
            FSM_state.MOVE_FORWARD: ('Moving forward.', MoveBase.Goal.MOVE_FORWARD),
            FSM_state.END: ('Ending the maze navigation.', MoveBase.Goal.STOP),
        }

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
        self.timer = self.create_timer(1 / self.update_frequency, self.state_machine)

        self.current_state = FSM_state.START
        self.next_state = FSM_state.MOVE_FORWARD

        # rate that determines the frequency of the loops
        self.loop_rate = self.create_rate(self.update_frequency, self.get_clock())

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
        self.current_speed = 0.0
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
            success = await self.move_forward(goal_handle, feedback)
        else:
            success = await self.rotate_in_place(goal_handle, feedback)

        result.success = success
        if success:
            goal_handle.succeed()
        else:
            goal_handle.canceled()
        return result

    async def move_forward(self, goal_handle, feedback):
        """
        Moves the robot forward with an acceleration ramp until the specified distance is traveled
        or an obstacle is detected within a safe range.
        """
        start_x, start_y = self.position
        distance_to_travel = 1.0  # meters
        acceleration = 0.1  # m/s²

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.stop_robot()
                return False

            current_x, current_y = self.position
            traveled = math.sqrt((current_x - start_x) ** 2 + (current_y - start_y) ** 2)

            if traveled >= distance_to_travel or self.front_distance <= self.distance_threshold / 2:
                break

            # Accelerate
            if self.current_speed < self.bot_max_lin_vel:
                self.current_speed += acceleration * 0.01
                self.current_speed = min(self.current_speed, self.bot_max_lin_vel)

            self.publish_cmd_vel(self.current_speed, 0.0)
            feedback.percentage_completed = min(traveled / distance_to_travel, 1.0) * 100.0
            goal_handle.publish_feedback(feedback)
            self.loop_rate.sleep()

        return True

    async def rotate_in_place(self, goal_handle, feedback):
        """
        Rotates the robot in place until the current yaw angle reaches the given target yaw angle
        within a specified tolerance.
        """
        start_yaw = self.orientation
        goal = goal_handle.request

        snapped_yaw = (math.pi / 2.0) * round(start_yaw / (math.pi / 2.0))
        delta = math.pi / 2.0 if goal.goal_move == goal.TURN_LEFT else -math.pi / 2.0
        target_angle = self.angle_diff(snapped_yaw + delta, start_yaw)
        goal_yaw = self.normalize_angle(start_yaw + target_angle)
        KP = 0.8
        ANGLE_TOLERANCE = 0.01

        self.stop_robot()

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.stop_robot()
                return False

            error = self.angle_diff(goal_yaw, self.orientation)

            if abs(error) < ANGLE_TOLERANCE:
                break

            ang_vel = KP * error
            ang_vel = max(min(ang_vel, self.bot_max_ang_vel), -self.bot_max_ang_vel)
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
        if future.result().result.success:
            self.action_complete = True

    def move_base_feedback_callback(self, feedback_msg: MoveBase.Feedback):
        """Callback for feedback."""
        pass

    def state_machine(self):
        """State machine to control the robot based on Hand On Wall algorithm"""
        if not self.should_execute_state_machine():
            return
        self.update_next_state()
        self.execute_state_action()
        self.current_state = self.next_state

    def should_execute_state_machine(self):
        return self.action_complete and self.current_state != FSM_state.END

    def update_next_state(self):
        """
        Updates the robot's next state based on its current position and sensor readings. Determines
        whether to end, start, turn, or move forward depending on sensor thresholds and the robot's
        current state.
        """
        x, y = self.position
        if y >= self.end_position_y:
            self.next_state = FSM_state.END
            return

        # Some states have almost identical transitions
        if self.current_state == FSM_state.START:
            if self.front_distance > self.distance_threshold:
                self.next_state = FSM_state.START
            else:
                self.next_state = FSM_state.TURN_RIGHT
            return

        if self.current_state in (FSM_state.TURN_RIGHT, FSM_state.TURN_LEFT):
            self.next_state = FSM_state.MOVE_FORWARD
            return

        if self.current_state == FSM_state.MOVE_FORWARD:
            if self.left_distance > self.distance_threshold:
                self.next_state = FSM_state.TURN_LEFT
            elif self.front_distance <= self.distance_threshold:
                self.next_state = FSM_state.TURN_RIGHT
            else:
                self.next_state = FSM_state.MOVE_FORWARD
            return

        if self.current_state == FSM_state.END:
            self.next_state = FSM_state.END

    def execute_state_action(self):
        log_message, goal_move = self.STATE_ACTION[self.next_state]
        self.get_logger().info(log_message)
        self.send_move_goal(MoveBase.Goal(goal_move=goal_move))


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

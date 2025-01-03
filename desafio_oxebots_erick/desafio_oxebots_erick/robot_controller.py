import math
from typing import Any, Optional, Union

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from std_srvs.srv import SetBool
from tf_transformations import euler_from_quaternion

from desafio_oxebots_erick_interfaces.action import MoveBase


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

        # sensor data
        self.front_distance = 0.0
        self.left_distance = 0.0
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
            Odometry, '/odom', self.odomCallback, qos_profile=10
        )
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laserCallback, qos_profile=10
        )
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imuCallback, qos_profile=10)

        # publisher to move the robot
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile=10)

        # action server for handling actions to move the robot
        self.move_base_server = ActionServer(
            self,
            MoveBase,
            '/move_base_action',
            execute_callback=self.moveBaseExecute,
            goal_callback=self.moveBaseHandleGoal,
            cancel_callback=self.moveBaseHandleCancel,
            handle_accepted_callback=self.moveBaseHandleAccepted,
        )

        # action client for sending action goals to move the robot
        self.move_base_client = ActionClient(self, MoveBase, '/move_base_action')

        # timer for repeatedly invoking a callback to publish messages
        self.timer = self.create_timer(1.0, self.timerCallback)

        # rate that determines the frequency of the loops
        self.loop_rate = self.create_rate(100, self.get_clock())

    #
    # Data processing callbacks
    #

    def odomCallback(self, msg: Odometry):
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

    def laserCallback(self, msg: LaserScan):
        """Extract the front and left distances from the laser sensor messages"""
        self.front_distance = msg.ranges[0]
        self.left_distance = msg.ranges[1]

    def imuCallback(self, msg: Imu):
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
    # MoveBase Action
    #
    def moveBaseHandleGoal(self, goal: MoveBase.Goal) -> GoalResponse:
        """Processes action goal requests"""
        self.get_logger().info('Received MoveBase goal request')
        return GoalResponse.ACCEPT

    def moveBaseHandleCancel(self, goal: MoveBase.Goal) -> CancelResponse:
        """Processes action cancel requests"""
        self.get_logger().info('Received request to cancel MoveBase goal')
        return CancelResponse.ACCEPT

    def moveBaseHandleAccepted(self, goal: MoveBase.Goal):
        """Processes accepted action goal requests"""

        # execute action in a separate thread to avoid blocking
        goal.execute()

    async def moveBaseExecute(self, goal_handle: MoveBase.Goal) -> MoveBase.Result:
        """
        Execute callback for the MoveBase action.
        This is an *asynchronous* function so we can 'await' inside.
        """

        self.get_logger().info('Executing MoveBase goal')

        # Goal is the requested movement (uint8)
        goal = goal_handle.request
        movement_type = goal.goal_move

        # Prepare the result and feedback message
        result = MoveBase.Result()
        feedback = MoveBase.Feedback()
        feedback.percentage_completed = 0.0

        # Stop command: no movement, immediately succeed
        if movement_type == goal.STOP:
            self.get_logger().info('STOP goal received.')
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
            target_distance = 1.0
        elif movement_type == goal.MOVE_BACKWARD:
            target_distance = -1.0
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
            direction = 1.0 if target_distance > 0 else -1.0

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
                    self.get_logger().info(
                        f'Front obstacle detected at {self.front_distance:.2f} m. Stopping forward motion.'
                    )
                    break

                # Publish velocity commands to move the robot
                self.publish_cmd_vel(direction * linear_speed, 0.0)

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
        self.get_logger().info('Goal succeeded')

        return result

    def timerCallback(self):
        """Processes timer triggers"""
        self.get_logger().info('Timer triggered')


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

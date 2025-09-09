// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <panda_push_control/joint_velocity_push_controller.h>

#include <cmath>

#include <controller_interface/controller_base.h>
#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>

namespace panda_push_control {

bool JointVelocityPushController::init(hardware_interface::RobotHW* robot_hw,
                                          ros::NodeHandle& node_handle) {

	sub_equilibrium_pose_ = node_handle.subscribe(
      "equilibrium_pose", 20, &JointVelocityPushController::equilibriumPoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

	// debug
  pub_cone_error_ = node_handle.advertise<geometry_msgs::PointStamped>("cone_task_error", 100);
  pub_position_error_ = node_handle.advertise<geometry_msgs::PointStamped>("position_task_error", 100);

  velocity_joint_interface_ = robot_hw->get<hardware_interface::VelocityJointInterface>();
  if (velocity_joint_interface_ == nullptr) {
    ROS_ERROR(
        "JointVelocityPushController: Error getting velocity joint interface from hardware!");
    return false;
  }

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("JointVelocityPushController: Could not get parameter arm_id");
    return false;
  }

  // if (!node_handle.getParam("safety_dq_scale_", safety_dq_scale_)) {
  //   ROS_ERROR("JointVelocityPushController: Could not get parameter pos_error_scale");
  //   return false;
  // }
  // if (safety_dq_scale_ > 1.) {
  //   ROS_ERROR("JointVelocityPushController: safety_dq_scale_ should not be greater than 1!");
  //   return false;
  // }

  if (!node_handle.getParam("min_ee_z_pos", min_ee_z_pos_)){
    ROS_ERROR(
        "JointVelocityPushController: Could not read parameter min_ee_z_pos");
    return false;
  }

  if (!node_handle.getParam("min_ee_xy_pos/x", min_ee_xy_pos_(0))){
    ROS_ERROR(
        "JointVelocityPushController: Could not read parameter min_ee_xy_pos/x");
    return false;
  }

  if (!node_handle.getParam("min_ee_xy_pos/y", min_ee_xy_pos_(1))){
    ROS_ERROR(
        "JointVelocityPushController: Could not read parameter min_ee_xy_pos/y");
    return false;
  }

  if (!node_handle.getParam("max_ee_xy_pos/x", max_ee_xy_pos_(0))){
    ROS_ERROR(
        "JointVelocityPushController: Could not read parameter max_ee_xy_pos/x");
    return false;
  }

  if (!node_handle.getParam("max_ee_xy_pos/y", max_ee_xy_pos_(1))){
    ROS_ERROR(
        "JointVelocityPushController: Could not read parameter max_ee_xy_pos/y");
    return false;
  }

  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names)) {
    ROS_ERROR("JointVelocityPushController: Could not parse joint names");
  }
  if (joint_names.size() != 7) {
    ROS_ERROR_STREAM("JointVelocityPushController: Wrong number of joint names, got "
                     << joint_names.size() << " instead of 7 names!");
    return false;
  }
  velocity_joint_handles_.resize(7);
  for (size_t i = 0; i < 7; ++i) {
    try {
      velocity_joint_handles_[i] = velocity_joint_interface_->getHandle(joint_names[i]);
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "JointVelocityPushController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

	auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "JointVelocityPushController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "JointVelocityPushController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "JointVelocityPushController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "JointVelocityPushController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

	position_d_.setZero();
  dq_d_.setZero();

  consider_x_cone_task_ = false;

  // position and velocity limits
  q_max_ << 2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973;
  q_min_ << -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973;

  dq_max_ << 2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61;
  dq_min_ = (-1) * dq_max_;

  server_.setCallback(boost::bind(&JointVelocityPushController::ReconfigureCallback, this, _1, _2));

  return true;
}

void JointVelocityPushController::ReconfigureCallback(panda_push_control::ControllerConfig &config, uint32_t level) {
  if (level == 0xFFFFFFFF) {
      config.min_height = min_ee_z_pos_;
  }
  
  min_ee_z_pos_ = config.min_height;
  lambda_ = config.lambda;
  safety_dq_scale_ = config.safety_dq_scale;
  if (safety_dq_scale_ > 1.) {
    ROS_ERROR("JointVelocityPushController: safety_dq_scale_ should not be greater than 1!");
    safety_dq_scale_ = 0.2;
  }

  ros::param::set("/joint_velocity_push_controller/min_ee_z_pos", min_ee_z_pos_);

  ROS_INFO_STREAM("Applied new clustering config! New z min: " << min_ee_z_pos_ << "; new lambda: " << lambda_);
  return;
}

void JointVelocityPushController::starting(const ros::Time& /* time */) {
  // compute initial velocity with jacobian and set x_attractor to initial configuration
  franka::RobotState initial_state = state_handle_->getRobotState();
  // get jacobian
  std::array<double, 42> jacobian_array = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  // convert to eigen
  Eigen::Matrix<double, 7, 1> q_init_tmp(initial_state.q.data());
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));

  // set equilibrium point to current state
  position_d_ = initial_transform.translation();

  Eigen::Matrix3d initial_rotation(initial_transform.rotation());
  cone_ref_vec_x_ << initial_rotation(0,0), initial_rotation(1,0), initial_rotation(2,0); // set to x-axis of current ee-frame
  cone_ref_vec_z_ << 0, 0, -1;

	q_init_ = q_init_tmp;
}

void JointVelocityPushController::update(const ros::Time& /* time */,
                                            const ros::Duration& period) {

  // get state variables
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 42> jacobian_array = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

  // convert to Eigen
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data()); // measured joint position
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data()); // measured joint velocity
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());

  // === errors ====
  // cone task errors
  Eigen::VectorXd conepos_error(4);
  int idx_z_task = 0;
  // cone tasks
  if (consider_x_cone_task_){
    // align x-axis of base frame with x-axis of ee frame
    ee_x_axis_ << transform(0,0), transform(1,0), transform(2,0); // axis end-effector frame; orientation w.r.t. base frame
    conepos_error(0) = cone_ref_vec_x_.transpose()*ee_x_axis_ - 1;
    conepos_error.conservativeResize(5);
    idx_z_task = 1;
  }
  // align z-axis of ee frame with z-axis*(-1) of base frame
  ee_z_axis_ << transform(0,2), transform(1,2), transform(2,2); // axis end-effector frame; orientation w.r.t. base frame
  conepos_error(idx_z_task) = cone_ref_vec_z_.transpose()*ee_z_axis_ - 1; 
  // position task error 
  conepos_error.bottomRows(3) = (position_d_ - position);// * safety_dq_scale_;

  // === Jacobians === 
  Eigen::MatrixXd J_conepos(conepos_error.size(),7);
  J_conepos.setZero();
  // position and rotation 
  Eigen::Matrix<double,3,7> J_pos(jacobian.topRows(3));
  Eigen::Matrix<double,3,7> J_rot(jacobian.bottomRows(3));
  // cone tasks
  if (consider_x_cone_task_){
    Eigen::Matrix<double,3,3> hat_x(vec2skewSymmetricMat(ee_x_axis_));
    J_conepos.block(0,6,1,1) = cone_ref_vec_x_.transpose() * hat_x * J_rot.block(0,6,3,1);
  }
  Eigen::Matrix<double,3,3> hat_z(vec2skewSymmetricMat(ee_z_axis_));
  J_conepos.block(idx_z_task,0,1,6) = cone_ref_vec_z_.transpose() * hat_z * J_rot.block(0,0,3,6);
  J_conepos.block(idx_z_task + 1,0,3,6) = J_pos.block(0,0,3,6);

  // compute and set desired joint velocity
	Eigen::VectorXd dq_conepos(7);
	Eigen::MatrixXd J_conepos_pinv;
	pseudoInverse(J_conepos, J_conepos_pinv, lambda_); 
	dq_d_ = J_conepos_pinv * conepos_error;

  // ensure position and velocity limits
  Eigen::Matrix<double, 7, 1> pos_aware_dq_min = dq_min_.cwiseMax(q_min_ - q);
  Eigen::Matrix<double, 7, 1> pos_aware_dq_max = dq_max_.cwiseMin(q_max_ - q);

  Eigen::Matrix<double, 7, 1> scales = (dq_d_.array() / pos_aware_dq_min.array()).cwiseMax(dq_d_.array() / pos_aware_dq_max.array());
  scales = scales.unaryExpr([](double v) { return std::isfinite(v)? v : 1.0; });
  dq_d_ = (safety_dq_scale_ * dq_d_) / std::max(1., scales.maxCoeff());
  
	for (size_t i = 0; i < 7; ++i) {
    velocity_joint_handles_[i].setCommand(dq_d_[i]);
  }

  // debug
  cone_task_error_msg.header.stamp = ros::Time::now();
  cone_task_error_msg.point.x = conepos_error(0,0);
  cone_task_error_msg.point.y = cone_task_error_msg.point.z = 0;
  pub_cone_error_.publish(cone_task_error_msg);

  position_task_error_msg.header.stamp = ros::Time::now();
  position_task_error_msg.point.x = conepos_error(1,0);
  position_task_error_msg.point.y = conepos_error(2,0);
  position_task_error_msg.point.z = conepos_error(3,0);
  pub_position_error_.publish(position_task_error_msg);
  cone_task_error_msg.header.stamp = ros::Time::now();
  cone_task_error_msg.point.x = conepos_error(0,0);
  cone_task_error_msg.point.y = cone_task_error_msg.point.z = 0;
  pub_cone_error_.publish(cone_task_error_msg);

  position_task_error_msg.header.stamp = ros::Time::now();
  position_task_error_msg.point.x = conepos_error(1,0);
  position_task_error_msg.point.y = conepos_error(2,0);
  position_task_error_msg.point.z = conepos_error(3,0);
  pub_position_error_.publish(position_task_error_msg);

}

Eigen::MatrixXd JointVelocityPushController::vstack(Eigen::MatrixXd A, Eigen::MatrixXd B){
  Eigen::MatrixXd C(A.rows()+B.rows(), A.cols());
  C << A,B;
  return C;
}

Eigen::Matrix<double, 3, 3> JointVelocityPushController::vec2skewSymmetricMat(Eigen::Vector3d vec){
  Eigen::Matrix<double, 3, 3> hat;
  hat << 0, -vec(2), vec(1), 
         vec(2), 0, -vec(0),
         -vec(1), vec(0), 0;
  return hat;
}

void JointVelocityPushController::equilibriumPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg){
	// desired pose w.r.t. base frame
  // position
  position_d_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  position_d_.topRows(2) = position_d_.topRows(2).cwiseMax(min_ee_xy_pos_).cwiseMin(max_ee_xy_pos_);
  if (position_d_(2) < min_ee_z_pos_){
    position_d_(2) = min_ee_z_pos_;
    ROS_DEBUG_STREAM_NAMED("JointVelocityPushController","Desired target z-pos is smaller than min z position.");
  }

  // orientation
  if (msg->header.frame_id == "z_task") consider_x_cone_task_ = false;
  else consider_x_cone_task_ = true;

  if (consider_x_cone_task_){
    orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, msg->pose.orientation.w;
    Eigen::Matrix3d rot_mat(orientation_d_target_.toRotationMatrix());

    cone_ref_vec_x_ << rot_mat(0,0), rot_mat(1,0), 0;
    if(abs(rot_mat(2,0)) > 1e-2){
      cone_ref_vec_x_.normalize();
      ROS_INFO_STREAM_NAMED("JointVelocityPushController","z-component of cone reference vector is not zero. Will use normalized projection on xy-plane of base frame: " << cone_ref_vec_x_.transpose());
    }
  }
}

void JointVelocityPushController::stopping(const ros::Time& /*time*/) {
  // WARNING: DO NOT SEND ZERO VELOCITIES HERE AS IN CASE OF ABORTING DURING MOTION
  // A JUMP TO ZERO WILL BE COMMANDED PUTTING HIGH LOADS ON THE ROBOT. LET THE DEFAULT
  // BUILT-IN STOPPING BEHAVIOR SLOW DOWN THE ROBOT.
}

}  // namespace panda_push_control

PLUGINLIB_EXPORT_CLASS(panda_push_control::JointVelocityPushController,
                       controller_interface::ControllerBase)

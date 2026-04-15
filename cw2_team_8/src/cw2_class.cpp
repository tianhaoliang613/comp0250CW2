/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#include <cw2_class.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <pcl/common/transforms.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/exceptions.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace
{
constexpr double kPi = 3.14159265358979323846;

double normalize_angle(double a)
{
  while (a > kPi) {
    a -= 2.0 * kPi;
  }
  while (a < -kPi) {
    a += 2.0 * kPi;
  }
  return a;
}

double angle_distance(double a, double b)
{
  return std::abs(normalize_angle(a - b));
}

double percentile_from_sorted(const std::vector<double> &vals, double q01)
{
  if (vals.empty()) {
    return 0.0;
  }
  const double q = std::clamp(q01, 0.0, 1.0);
  const std::size_t idx = static_cast<std::size_t>(q * static_cast<double>(vals.size() - 1));
  return vals[idx];
}

bool task1_nought_cell_occupied(int ix, int iy)
{
  return std::abs(ix) == 2 || std::abs(iy) == 2;
}

bool task1_cross_cell_occupied(int ix, int iy)
{
  return ix == 0 || iy == 0;
}

double task1_point_to_rect_distance(double px, double py, double cx, double cy, double half_x, double half_y)
{
  const double dx = std::max(std::abs(px - cx) - half_x, 0.0);
  const double dy = std::max(std::abs(py - cy) - half_y, 0.0);
  return std::hypot(dx, dy);
}

double task1_template_fit_cost(
  const PointC &surface,
  const geometry_msgs::msg::Point &center,
  bool is_nought,
  bool is_cross,
  double yaw,
  double cell_size)
{
  std::vector<double> dists;
  dists.reserve(surface.size());
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  const double half = 0.5 * cell_size;

  for (const auto &p : surface.points) {
    const double dx = p.x - center.x;
    const double dy = p.y - center.y;
    const double lx = c * dx + s * dy;
    const double ly = -s * dx + c * dy;

    double best = std::numeric_limits<double>::infinity();
    for (int ix = -2; ix <= 2; ++ix) {
      for (int iy = -2; iy <= 2; ++iy) {
        const bool occupied =
          (is_nought && task1_nought_cell_occupied(ix, iy)) ||
          (is_cross && task1_cross_cell_occupied(ix, iy));
        if (!occupied) {
          continue;
        }
        const double cx = static_cast<double>(ix) * cell_size;
        const double cy = static_cast<double>(iy) * cell_size;
        best = std::min(best, task1_point_to_rect_distance(lx, ly, cx, cy, half, half));
      }
    }
    if (std::isfinite(best)) {
      dists.push_back(best);
    }
  }

  if (dists.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  std::sort(dists.begin(), dists.end());
  return percentile_from_sorted(dists, 0.78);
}

Eigen::Affine3f affine_from_transform_msg(const geometry_msgs::msg::Transform &t)
{
  Eigen::Translation3f tr(
    static_cast<float>(t.translation.x),
    static_cast<float>(t.translation.y),
    static_cast<float>(t.translation.z));
  Eigen::Quaternionf q(
    static_cast<float>(t.rotation.w),
    static_cast<float>(t.rotation.x),
    static_cast<float>(t.rotation.y),
    static_cast<float>(t.rotation.z));
  return Eigen::Affine3f(tr * q);
}
}  // namespace

cw2::cw2(const rclcpp::Node::SharedPtr &node)
: node_(node),
  tf_buffer_(node->get_clock()),
  tf_listener_(tf_buffer_),
  g_cloud_ptr(new PointC)
{
  task_service_callback_group_ =
    node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  t1_service_ = node_->create_service<cw2_world_spawner::srv::Task1Service>(
    "/task1_start",
    std::bind(&cw2::t1_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default,
    task_service_callback_group_);
  t2_service_ = node_->create_service<cw2_world_spawner::srv::Task2Service>(
    "/task2_start",
    std::bind(&cw2::t2_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default,
    task_service_callback_group_);
  t3_service_ = node_->create_service<cw2_world_spawner::srv::Task3Service>(
    "/task3_start",
    std::bind(&cw2::t3_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default,
    task_service_callback_group_);

  pointcloud_topic_ = node_->declare_parameter<std::string>(
    "pointcloud_topic", "/r200/camera/depth_registered/points");
  pointcloud_qos_reliable_ =
    node_->declare_parameter<bool>("pointcloud_qos_reliable", true);
  task1_debug_verbose_ = node_->declare_parameter<bool>("task1_debug_verbose", task1_debug_verbose_);

  task1_cloud_max_age_sec_ = node_->declare_parameter<double>(
    "task1_cloud_max_age_sec", task1_cloud_max_age_sec_);
  task1_cloud_wait_timeout_sec_ = node_->declare_parameter<double>(
    "task1_cloud_wait_timeout_sec", task1_cloud_wait_timeout_sec_);
  task1_spawn_settle_ms_ = node_->declare_parameter<int>("task1_spawn_settle_ms", task1_spawn_settle_ms_);

  task1_roi_radius_xy_ = node_->declare_parameter<double>("task1_roi_radius_xy", task1_roi_radius_xy_);
  task1_roi_half_height_ =
    node_->declare_parameter<double>("task1_roi_half_height", task1_roi_half_height_);
  task1_surface_percentile_ =
    node_->declare_parameter<double>("task1_surface_percentile", task1_surface_percentile_);
  task1_ground_reject_height_ =
    node_->declare_parameter<double>("task1_ground_reject_height", task1_ground_reject_height_);

  task1_gripper_open_width_ =
    node_->declare_parameter<double>("task1_gripper_open_width", task1_gripper_open_width_);
  task1_gripper_max_width_ =
    node_->declare_parameter<double>("task1_gripper_max_width", task1_gripper_max_width_);
  task1_gripper_min_width_ =
    node_->declare_parameter<double>("task1_gripper_min_width", task1_gripper_min_width_);
  task1_gripper_closing_margin_ = node_->declare_parameter<double>(
    "task1_gripper_closing_margin", task1_gripper_closing_margin_);

  task1_candidate_strip_half_depth_ = node_->declare_parameter<double>(
    "task1_candidate_strip_half_depth", task1_candidate_strip_half_depth_);
  task1_contact_side_threshold_ = node_->declare_parameter<double>(
    "task1_contact_side_threshold", task1_contact_side_threshold_);
  task1_min_contact_points_per_side_ = node_->declare_parameter<int>(
    "task1_min_contact_points_per_side", task1_min_contact_points_per_side_);
  task1_max_candidates_to_try_ = node_->declare_parameter<int>(
    "task1_max_candidates_to_try", task1_max_candidates_to_try_);
  task1_n_yaw_samples_ = node_->declare_parameter<int>("task1_n_yaw_samples", task1_n_yaw_samples_);

  task1_pregrasp_offset_z_ =
    node_->declare_parameter<double>("task1_pregrasp_offset_z", task1_pregrasp_offset_z_);
  task1_grasp_offset_z_ =
    node_->declare_parameter<double>("task1_grasp_offset_z", task1_grasp_offset_z_);
  task1_lift_offset_z_ = node_->declare_parameter<double>("task1_lift_offset_z", task1_lift_offset_z_);
  task1_place_offset_z_ = node_->declare_parameter<double>("task1_place_offset_z", task1_place_offset_z_);
  task1_release_min_extra_z_ = node_->declare_parameter<double>(
    "task1_release_min_extra_z", task1_release_min_extra_z_);
  task1_release_max_extra_z_ = node_->declare_parameter<double>(
    "task1_release_max_extra_z", task1_release_max_extra_z_);
  task1_release_object_height_scale_ = node_->declare_parameter<double>(
    "task1_release_object_height_scale", task1_release_object_height_scale_);

  task1_basket_approach_min_z_ =
    node_->declare_parameter<double>("task1_basket_approach_min_z", task1_basket_approach_min_z_);
  task1_transit_raise_above_lift_ =
    node_->declare_parameter<double>("task1_transit_raise_above_lift", task1_transit_raise_above_lift_);
  task1_transit_xy_vel_scale_ =
    node_->declare_parameter<double>("task1_transit_xy_vel_scale", task1_transit_xy_vel_scale_);
  task1_transit_xy_acc_scale_ =
    node_->declare_parameter<double>("task1_transit_xy_acc_scale", task1_transit_xy_acc_scale_);
  task1_place_descend_vel_scale_ =
    node_->declare_parameter<double>("task1_place_descend_vel_scale", task1_place_descend_vel_scale_);
  task1_place_descend_acc_scale_ =
    node_->declare_parameter<double>("task1_place_descend_acc_scale", task1_place_descend_acc_scale_);
  task1_ee_yaw_offset_rad_ =
    node_->declare_parameter<double>("task1_ee_yaw_offset_rad", task1_ee_yaw_offset_rad_);

  task1_surface_z_percentile_ = node_->declare_parameter<double>(
    "task1_surface_z_percentile", task1_surface_z_percentile_);
  task1_tcp_clearance_above_surface_ = node_->declare_parameter<double>(
    "task1_tcp_clearance_above_surface", task1_tcp_clearance_above_surface_);
  task1_descend_velocity_scale_ = node_->declare_parameter<double>(
    "task1_descend_velocity_scale", task1_descend_velocity_scale_);
  task1_descend_acceleration_scale_ = node_->declare_parameter<double>(
    "task1_descend_acceleration_scale", task1_descend_acceleration_scale_);
  task1_approach_last_delta_z_ = node_->declare_parameter<double>(
    "task1_approach_last_delta_z", task1_approach_last_delta_z_);
  task1_grasp_settle_ms_ = node_->declare_parameter<int>("task1_grasp_settle_ms", task1_grasp_settle_ms_);
  task1_grasp_backoff_z_ = node_->declare_parameter<double>("task1_grasp_backoff_z", task1_grasp_backoff_z_);
  task1_blocked_close_width_tolerance_ = node_->declare_parameter<double>(
    "task1_blocked_close_width_tolerance", task1_blocked_close_width_tolerance_);

  pointcloud_callback_group_ =
    node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions pointcloud_sub_options;
  pointcloud_sub_options.callback_group = pointcloud_callback_group_;

  rclcpp::QoS pointcloud_qos = rclcpp::SensorDataQoS();
  if (pointcloud_qos_reliable_) {
    pointcloud_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
  }

  color_cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
    pointcloud_topic_,
    pointcloud_qos,
    std::bind(&cw2::cloud_callback, this, std::placeholders::_1),
    pointcloud_sub_options);

  arm_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "panda_arm");
  hand_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "hand");
  setup_arm_hand(*arm_group_, *hand_group_);

  RCLCPP_INFO(
    node_->get_logger(),
    "cw2_team_8: Task1 pick-place ready. Cloud topic '%s' (%s QoS) arm_pose_ref=%s planning_frame=%s",
    pointcloud_topic_.c_str(),
    pointcloud_qos_reliable_ ? "reliable" : "sensor-data",
    arm_group_->getPoseReferenceFrame().c_str(),
    arm_group_->getPlanningFrame().c_str());
}

void cw2::setup_arm_hand(
  moveit::planning_interface::MoveGroupInterface &arm,
  moveit::planning_interface::MoveGroupInterface &hand) const
{
  // Task geometry is computed in panda_link0 (service points + TF cloud). MoveGroup defaults to
  // planning_frame=world — without this, pose targets are misinterpreted (seen in logs as wrong z).
  arm.setPoseReferenceFrame("panda_link0");

  arm.setPlanningTime(12.0);
  arm.setNumPlanningAttempts(8);
  arm.setMaxVelocityScalingFactor(0.35);
  arm.setMaxAccelerationScalingFactor(0.35);
  arm.setGoalTolerance(0.002);
  arm.setPlannerId("RRTConnect");

  hand.setPlanningTime(5.0);
  hand.setNumPlanningAttempts(5);
  hand.setMaxVelocityScalingFactor(0.5);
  hand.setMaxAccelerationScalingFactor(0.5);
}

void cw2::cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  pcl::PCLPointCloud2 pcl_cloud;
  pcl_conversions::toPCL(*msg, pcl_cloud);

  PointCPtr latest_cloud(new PointC);
  pcl::fromPCLPointCloud2(pcl_cloud, *latest_cloud);

  std::lock_guard<std::mutex> lock(cloud_mutex_);
  g_input_pc_frame_id_ = msg->header.frame_id;
  g_cloud_ptr = std::move(latest_cloud);
  ++g_cloud_sequence_;
  g_cloud_receive_steady_ = std::chrono::steady_clock::now();
}

bool cw2::wait_for_cloud(
  std::chrono::milliseconds timeout,
  std::uint64_t min_sequence_exclusive,
  std::uint64_t *latest_sequence) const
{
  const auto start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start < timeout) {
    std::uint64_t seq = 0;
    std::chrono::steady_clock::time_point receive_tp = std::chrono::steady_clock::time_point::min();
    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      seq = g_cloud_sequence_;
      receive_tp = g_cloud_receive_steady_;
    }
    if (seq > min_sequence_exclusive && receive_tp != std::chrono::steady_clock::time_point::min()) {
      const auto age = std::chrono::steady_clock::now() - receive_tp;
      const auto age_limit = std::chrono::duration<double>(std::max(0.05, task1_cloud_max_age_sec_));
      if (age > age_limit) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }
      if (latest_sequence != nullptr) {
        *latest_sequence = seq;
      }
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return false;
}

bool cw2::task1_cloud_recently_valid(std::chrono::seconds max_age) const
{
  std::lock_guard<std::mutex> lock(cloud_mutex_);
  if (!g_cloud_ptr || g_cloud_ptr->empty()) {
    return false;
  }
  if (g_cloud_receive_steady_ == std::chrono::steady_clock::time_point::min()) {
    return false;
  }
  const auto age = std::chrono::steady_clock::now() - g_cloud_receive_steady_;
  return age <= max_age;
}

bool cw2::get_cloud_snapshot(PointCPtr &cloud, std::string &frame_id, std::uint64_t &sequence) const
{
  std::lock_guard<std::mutex> lock(cloud_mutex_);
  if (!g_cloud_ptr || g_cloud_ptr->empty()) {
    return false;
  }
  cloud.reset(new PointC(*g_cloud_ptr));
  frame_id = g_input_pc_frame_id_;
  sequence = g_cloud_sequence_;
  return true;
}

bool cw2::transform_point_to_link0(
  const geometry_msgs::msg::PointStamped &in,
  geometry_msgs::msg::PointStamped &out) const
{
  if (in.header.frame_id.empty() || in.header.frame_id == "panda_link0") {
    out = in;
    out.header.frame_id = "panda_link0";
    out.header.stamp = node_->get_clock()->now();
    return true;
  }
  try {
    geometry_msgs::msg::TransformStamped tf_msg = tf_buffer_.lookupTransform(
      "panda_link0", in.header.frame_id, tf2::TimePointZero, tf2::durationFromSec(3.0));
    tf2::doTransform(in, out, tf_msg);
    out.header.frame_id = "panda_link0";
    out.header.stamp = node_->get_clock()->now();
    return true;
  } catch (const tf2::TransformException &ex) {
    RCLCPP_ERROR(node_->get_logger(), "TF point->panda_link0 failed: %s", ex.what());
    return false;
  }
}

bool cw2::transform_cloud_to_frame(
  const PointC &cloud_in,
  const std::string &input_frame,
  const std::string &target_frame,
  PointC &cloud_out) const
{
  if (input_frame == target_frame) {
    cloud_out = cloud_in;
    return true;
  }
  try {
    geometry_msgs::msg::TransformStamped tf_msg = tf_buffer_.lookupTransform(
      target_frame, input_frame, tf2::TimePointZero, tf2::durationFromSec(3.0));
    const Eigen::Affine3f t = affine_from_transform_msg(tf_msg.transform);
    pcl::transformPointCloud(cloud_in, cloud_out, t);
    return true;
  } catch (const tf2::TransformException &ex) {
    RCLCPP_ERROR(
      node_->get_logger(), "TF cloud %s -> %s failed: %s", input_frame.c_str(), target_frame.c_str(),
      ex.what());
    return false;
  }
}

void cw2::extract_task1_roi_points(
  const PointC &cloud_in_link0,
  const geometry_msgs::msg::Point &object_point_link0,
  PointC &roi_out) const
{
  roi_out.clear();
  roi_out.header = cloud_in_link0.header;
  const double ox = object_point_link0.x;
  const double oy = object_point_link0.y;
  const double oz = object_point_link0.z;
  const double rxy = task1_roi_radius_xy_;
  const double hz = task1_roi_half_height_;

  for (const auto &p : cloud_in_link0.points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
      continue;
    }
    const double dx = p.x - ox;
    const double dy = p.y - oy;
    const double dz = p.z - oz;
    if (dx * dx + dy * dy > rxy * rxy) {
      continue;
    }
    if (std::abs(dz) > hz) {
      continue;
    }
    if (p.z < task1_ground_reject_height_) {
      continue;
    }
    roi_out.push_back(p);
  }
}

void cw2::generate_task1_candidates(
  const PointC &roi,
  const geometry_msgs::msg::Point &object_point_link0,
  const std::string &shape_type,
  std::vector<Task1GraspCandidate> &candidates_out) const
{
  candidates_out.clear();

  const bool is_nought = shape_type.find("nought") != std::string::npos;
  const bool is_cross = shape_type.find("cross") != std::string::npos;
  if (!is_nought && !is_cross) {
    return;
  }

  // ===== OFFLINE PHASE (Lecture 7): predefined grasp hypotheses per known object model =====
  // Task1 objects: x = 40mm, 5x5 grid, height 40mm.
  // Hypotheses are defined in the object's local frame (origin = centroid, yaw = 0).
  // Each hypothesis: (dx, dy, grasp_yaw_in_object_frame, gripper_close_width)
  struct LocalHypothesis {
    double dx;
    double dy;
    double yaw;
    double close_width;
  };

  constexpr double kCell = 0.040;
  constexpr double kRingThickness = kCell;
  constexpr double kRingSideOffset = 2.0 * kCell;
  constexpr double kCrossArmHalfWidth = 0.5 * kCell;
  constexpr double kCrossGrabOffset = 1.5 * kCell;

  std::vector<LocalHypothesis> hypotheses;

  if (is_nought) {
    const double w = std::clamp(kRingThickness - task1_gripper_closing_margin_,
                                task1_gripper_min_width_, task1_gripper_max_width_);
    hypotheses.push_back({ kRingSideOffset,  0.0,             0.0,            w});
    hypotheses.push_back({-kRingSideOffset,  0.0,             kPi,            w});
    hypotheses.push_back({ 0.0,              kRingSideOffset,  0.5 * kPi,     w});
    hypotheses.push_back({ 0.0,             -kRingSideOffset, -0.5 * kPi,     w});
  } else {
    const double w = std::clamp(2.0 * kCrossArmHalfWidth - task1_gripper_closing_margin_,
                                task1_gripper_min_width_, task1_gripper_max_width_);
    hypotheses.push_back({ kCrossGrabOffset, 0.0,  0.5 * kPi, w});
    hypotheses.push_back({-kCrossGrabOffset, 0.0,  0.5 * kPi, w});
    hypotheses.push_back({0.0,  kCrossGrabOffset,  0.0,        w});
    hypotheses.push_back({0.0, -kCrossGrabOffset,  0.0,        w});
  }

  // ===== ONLINE PHASE (Lecture 7): segment scene, estimate object pose, retrieve hypotheses =====

  // Step 1: Extract surface points for pose (yaw) estimation.
  PointC surface;
  if (!roi.empty()) {
    std::vector<double> zs;
    zs.reserve(roi.size());
    for (const auto &p : roi.points) {
      zs.push_back(p.z);
    }
    std::sort(zs.begin(), zs.end());
    const double z_thr = percentile_from_sorted(zs, task1_surface_percentile_);
    surface.header = roi.header;
    for (const auto &p : roi.points) {
      if (p.z >= z_thr) {
        surface.push_back(p);
      }
    }
    if (surface.size() < 8U) {
      surface = roi;
    }
  }

  // Step 2: Estimate object yaw by template fitting (pose estimation).
  double object_yaw = 0.0;
  if (surface.size() >= 8U) {
    double best_cost = std::numeric_limits<double>::infinity();
    const int n_yaw = std::max(36, task1_n_yaw_samples_ * 3);
    for (int i = 0; i < n_yaw; ++i) {
      const double yaw = (0.5 * kPi) * static_cast<double>(i) / static_cast<double>(n_yaw);
      const double cost = task1_template_fit_cost(
        surface, object_point_link0, is_nought, is_cross, yaw, kCell);
      if (cost < best_cost) {
        best_cost = cost;
        object_yaw = yaw;
      }
    }
    for (double delta : {0.0, 0.02, -0.02, 0.04, -0.04, 0.08, -0.08}) {
      const double yaw = normalize_angle(object_yaw + delta);
      const double cost = task1_template_fit_cost(
        surface, object_point_link0, is_nought, is_cross, yaw, kCell);
      if (cost < best_cost) {
        best_cost = cost;
        object_yaw = yaw;
      }
    }
    if (task1_debug_verbose_) {
      RCLCPP_INFO(node_->get_logger(),
        "[T1DBG] pose_estimation shape=%s object_yaw=%.1fdeg fit_cost=%.4f surface_pts=%zu",
        shape_type.c_str(), object_yaw * 180.0 / kPi, best_cost, surface.size());
    }
  }

  // Step 3: Transform offline hypotheses from object frame to panda_link0 (retrieve).
  const double cy = std::cos(object_yaw);
  const double sy = std::sin(object_yaw);

  for (std::size_t i = 0; i < hypotheses.size(); ++i) {
    const auto &h = hypotheses[i];
    Task1GraspCandidate cand;
    cand.x = object_point_link0.x + cy * h.dx - sy * h.dy;
    cand.y = object_point_link0.y + sy * h.dx + cy * h.dy;
    cand.yaw = normalize_angle(h.yaw + object_yaw);
    cand.close_width = h.close_width;
    cand.score = static_cast<double>(hypotheses.size() - i);
    candidates_out.push_back(cand);
  }

  if (task1_debug_verbose_) {
    RCLCPP_INFO(node_->get_logger(),
      "[T1DBG] retrieved %zu grasp hypotheses for shape=%s",
      candidates_out.size(), shape_type.c_str());
    for (std::size_t i = 0; i < candidates_out.size(); ++i) {
      const auto &c = candidates_out[i];
      RCLCPP_INFO(node_->get_logger(),
        "[T1DBG]   hyp[%zu] xy=(%.4f,%.4f) yaw=%.1fdeg close=%.4f",
        i, c.x, c.y, c.yaw * 180.0 / kPi, c.close_width);
    }
  }
}

double cw2::task1_arm_link8_yaw(double grasp_model_yaw_rad) const
{
  return normalize_angle(grasp_model_yaw_rad + task1_ee_yaw_offset_rad_);
}

geometry_msgs::msg::Pose cw2::make_topdown_pose(double x, double y, double z, double yaw_rad) const
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  tf2::Quaternion q;
  q.setRPY(0.0, kPi, yaw_rad);
  pose.orientation = tf2::toMsg(q);
  return pose;
}

bool cw2::execute_plan(
  moveit::planning_interface::MoveGroupInterface &group,
  const moveit::planning_interface::MoveGroupInterface::Plan &plan) const
{
  return group.execute(plan) == moveit::core::MoveItErrorCode::SUCCESS;
}

bool cw2::move_arm_to_pose(const geometry_msgs::msg::Pose &target_pose) const
{
  arm_group_->setStartStateToCurrentState();
  arm_group_->setPoseTarget(target_pose);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  const auto pcode = arm_group_->plan(plan);
  if (pcode != moveit::core::MoveItErrorCode::SUCCESS) {
    if (task1_debug_verbose_) {
      RCLCPP_WARN(
        node_->get_logger(),
        "[T1DBG] move_arm_to_pose plan FAILED code=%d pos=(%.4f,%.4f,%.4f) quat=(%.3f,%.3f,%.3f,%.3f) frame=%s",
        pcode.val, target_pose.position.x, target_pose.position.y, target_pose.position.z,
        target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z,
        target_pose.orientation.w, arm_group_->getPlanningFrame().c_str());
    }
    return false;
  }
  const bool ok = execute_plan(*arm_group_, plan);
  if (!ok && task1_debug_verbose_) {
    RCLCPP_WARN(node_->get_logger(), "[T1DBG] move_arm_to_pose execute FAILED");
  }
  return ok;
}

bool cw2::cartesian_move(const geometry_msgs::msg::Pose &target_pose, double min_fraction) const
{
  arm_group_->setStartStateToCurrentState();
  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(target_pose);
  moveit_msgs::msg::RobotTrajectory traj;
  const double fraction = arm_group_->computeCartesianPath(waypoints, 0.002, 0.0, traj, false);
  if (task1_debug_verbose_) {
    RCLCPP_INFO(
      node_->get_logger(),
      "[T1DBG] cartesian_move 1wp fraction=%.3f need>=%.3f target_z=%.4f traj_pts=%zu", fraction, min_fraction,
      target_pose.position.z, traj.joint_trajectory.points.size());
  }
  if (fraction < min_fraction) {
    return false;
  }
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  plan.trajectory_ = traj;
  const bool ok = arm_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS;
  if (!ok && task1_debug_verbose_) {
    RCLCPP_WARN(node_->get_logger(), "[T1DBG] cartesian_move execute FAILED (fraction was ok)");
  }
  return ok;
}

double cw2::task1_percentile_z(const PointC &roi, double q01)
{
  if (roi.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::vector<double> zs;
  zs.reserve(roi.size());
  for (const auto &p : roi.points) {
    if (std::isfinite(p.z)) {
      zs.push_back(p.z);
    }
  }
  if (zs.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(zs.begin(), zs.end());
  const double q = std::clamp(q01, 0.0, 1.0);
  const std::size_t idx = static_cast<std::size_t>(q * static_cast<double>(zs.size() - 1));
  return zs[idx];
}

bool cw2::cartesian_follow_waypoints(
  const std::vector<geometry_msgs::msg::Pose> &waypoints,
  double min_fraction,
  double vel_scale,
  double acc_scale,
  double eef_step) const
{
  if (waypoints.empty()) {
    return false;
  }
  arm_group_->setMaxVelocityScalingFactor(vel_scale);
  arm_group_->setMaxAccelerationScalingFactor(acc_scale);
  arm_group_->setStartStateToCurrentState();
  moveit_msgs::msg::RobotTrajectory traj;
  const double fraction = arm_group_->computeCartesianPath(waypoints, eef_step, 0.0, traj, false);
  arm_group_->setMaxVelocityScalingFactor(0.35);
  arm_group_->setMaxAccelerationScalingFactor(0.35);
  if (task1_debug_verbose_) {
    const double z0 = waypoints.front().position.z;
    const double z1 = waypoints.back().position.z;
    RCLCPP_INFO(
      node_->get_logger(),
      "[T1DBG] cartesian_follow n_wp=%zu fraction=%.3f need>=%.3f vel=%.2f acc=%.2f step=%.4f z_first=%.4f z_last=%.4f "
      "traj_pts=%zu",
      waypoints.size(), fraction, min_fraction, vel_scale, acc_scale, eef_step, z0, z1,
      traj.joint_trajectory.points.size());
  }
  if (fraction < min_fraction) {
    return false;
  }
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  plan.trajectory_ = traj;
  const bool ok = arm_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS;
  if (!ok && task1_debug_verbose_) {
    RCLCPP_WARN(node_->get_logger(), "[T1DBG] cartesian_follow execute FAILED");
  }
  return ok;
}

bool cw2::set_gripper(double width_m) const
{
  const double w = std::clamp(width_m, 0.0, task1_gripper_open_width_);
  hand_group_->setStartStateToCurrentState();
  hand_group_->setJointValueTarget("panda_finger_joint1", w / 2.0);
  hand_group_->setJointValueTarget("panda_finger_joint2", w / 2.0);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  auto pcode = hand_group_->plan(plan);
  if (pcode != moveit::core::MoveItErrorCode::SUCCESS) {
    if (task1_debug_verbose_) {
      RCLCPP_WARN(
        node_->get_logger(), "[T1DBG] set_gripper plan FAILED code=%d width=%.4f — retry +0.4mm", pcode.val, w);
    }
    const double w2 = std::clamp(w + 0.0008, 0.0, task1_gripper_open_width_);
    hand_group_->setStartStateToCurrentState();
    hand_group_->setJointValueTarget("panda_finger_joint1", w2 / 2.0);
    hand_group_->setJointValueTarget("panda_finger_joint2", w2 / 2.0);
    pcode = hand_group_->plan(plan);
    if (pcode != moveit::core::MoveItErrorCode::SUCCESS) {
      return false;
    }
  }
  const bool ok = execute_plan(*hand_group_, plan);
  if (task1_debug_verbose_) {
    RCLCPP_INFO(node_->get_logger(), "[T1DBG] set_gripper width=%.4f per_joint=%.4f ok=%d", w, w / 2.0, ok ? 1 : 0);
  }
  return ok;
}

double cw2::current_gripper_width() const
{
  const auto vals = hand_group_->getCurrentJointValues();
  if (vals.size() >= 2U && std::isfinite(vals[0]) && std::isfinite(vals[1])) {
    return std::max(0.0, vals[0] + vals[1]);
  }
  return task1_gripper_open_width_;
}

bool cw2::close_gripper_for_grasp(double target_width_m) const
{
  const double target = std::clamp(target_width_m, 0.0, task1_gripper_open_width_);
  if (set_gripper(target)) {
    return true;
  }

  // In Gazebo the hand action often aborts when the fingers are blocked by the object.
  // Treat that as a successful grasp if the hand actually closed enough to indicate contact.
  std::this_thread::sleep_for(std::chrono::milliseconds(80));
  const double achieved = current_gripper_width();
  const bool blocked_with_contact =
    achieved <= std::min(task1_gripper_open_width_ - 0.006, target + task1_blocked_close_width_tolerance_);
  if (task1_debug_verbose_) {
    RCLCPP_INFO(
      node_->get_logger(),
      "[T1DBG] close_gripper_for_grasp target=%.4f achieved=%.4f blocked_contact=%d",
      target, achieved, blocked_with_contact ? 1 : 0);
  }
  return blocked_with_contact;
}

void cw2::task1_debug_roi_stats(const PointC &roi, const geometry_msgs::msg::Point &obj) const
{
  if (!task1_debug_verbose_ || roi.empty()) {
    return;
  }
  double zmin = std::numeric_limits<double>::infinity();
  double zmax = -std::numeric_limits<double>::infinity();
  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
  std::size_t n = 0;
  for (const auto &p : roi.points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
      continue;
    }
    zmin = std::min(zmin, static_cast<double>(p.z));
    zmax = std::max(zmax, static_cast<double>(p.z));
    sx += p.x;
    sy += p.y;
    sz += p.z;
    ++n;
  }
  if (n == 0U) {
    RCLCPP_WARN(node_->get_logger(), "[T1DBG] roi stats: no finite points");
    return;
  }
  sx /= static_cast<double>(n);
  sy /= static_cast<double>(n);
  sz /= static_cast<double>(n);
  RCLCPP_INFO(
    node_->get_logger(),
    "[T1DBG] roi pts=%zu z[min,max]=[%.4f,%.4f] mean=(%.4f,%.4f,%.4f) obj=(%.4f,%.4f,%.4f) dz_mean_obj=%.4f",
    roi.size(), zmin, zmax, sx, sy, sz, obj.x, obj.y, obj.z, sz - obj.z);
}

void cw2::t1_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response)
{
  (void)response;

  geometry_msgs::msg::PointStamped obj_l0;
  geometry_msgs::msg::PointStamped goal_l0;
  if (!transform_point_to_link0(request->object_point, obj_l0)) {
    RCLCPP_ERROR(node_->get_logger(), "Task1: failed to transform object_point to panda_link0");
    return;
  }
  if (!transform_point_to_link0(request->goal_point, goal_l0)) {
    RCLCPP_ERROR(node_->get_logger(), "Task1: failed to transform goal_point to panda_link0");
    return;
  }

  RCLCPP_INFO(
    node_->get_logger(),
    "Task1 start: shape=%s object=(%.3f,%.3f,%.3f) goal=(%.3f,%.3f,%.3f) (panda_link0)",
    request->shape_type.c_str(), obj_l0.point.x, obj_l0.point.y, obj_l0.point.z, goal_l0.point.x,
    goal_l0.point.y, goal_l0.point.z);

  if (task1_spawn_settle_ms_ > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(task1_spawn_settle_ms_));
  }

  std::uint64_t seq_before = 0;
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    seq_before = g_cloud_sequence_;
  }
  const auto wait_ms = std::chrono::milliseconds(
    static_cast<int>(std::max(1.0, task1_cloud_wait_timeout_sec_) * 1000.0));
  if (!wait_for_cloud(wait_ms, seq_before, nullptr)) {
    if (task1_cloud_recently_valid(std::chrono::seconds(3))) {
      RCLCPP_WARN(
        node_->get_logger(),
        "Task1: no new point-cloud sequence within timeout (seq>%llu); using latest cloud for repeat/low-rate "
        "camera",
        static_cast<unsigned long long>(seq_before));
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Task1: no fresh point cloud within timeout");
      return;
    }
  }

  PointCPtr cloud_cam;
  std::string cloud_frame;
  std::uint64_t cloud_seq = 0;
  if (!get_cloud_snapshot(cloud_cam, cloud_frame, cloud_seq) || !cloud_cam || cloud_cam->empty()) {
    RCLCPP_ERROR(node_->get_logger(), "Task1: empty cloud snapshot");
    return;
  }

  PointC cloud_l0;
  if (!transform_cloud_to_frame(*cloud_cam, cloud_frame, "panda_link0", cloud_l0)) {
    RCLCPP_ERROR(node_->get_logger(), "Task1: cloud TF to panda_link0 failed");
    return;
  }

  if (task1_debug_verbose_) {
    RCLCPP_INFO(
      node_->get_logger(),
      "[T1DBG] cloud cam_pts=%zu frame='%s' seq=%llu -> link0_pts=%zu planning_frame=%s eef_link=%s",
      cloud_cam->size(), cloud_frame.c_str(), static_cast<unsigned long long>(cloud_seq), cloud_l0.size(),
      arm_group_->getPlanningFrame().c_str(), arm_group_->getEndEffectorLink().c_str());
  }

  PointC roi;
  extract_task1_roi_points(cloud_l0, obj_l0.point, roi);
  if (roi.size() < 40U) {
    RCLCPP_ERROR(
      node_->get_logger(), "Task1: insufficient ROI points (%zu) — widen ROI parameters if needed",
      roi.size());
    return;
  }

  task1_debug_roi_stats(roi, obj_l0.point);

  std::vector<Task1GraspCandidate> candidates;
  generate_task1_candidates(roi, obj_l0.point, request->shape_type, candidates);
  if (candidates.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "Task1: no valid grasp candidates");
    return;
  }

  if (task1_debug_verbose_) {
    const int show = std::min(5, static_cast<int>(candidates.size()));
    for (int i = 0; i < show; ++i) {
      const auto &cc = candidates[static_cast<std::size_t>(i)];
      RCLCPP_INFO(
        node_->get_logger(), "[T1DBG] cand[%d] score=%.2f xy=(%.4f,%.4f) yaw_deg=%.1f close=%.4f", i,
        cc.score, cc.x, cc.y, cc.yaw * 180.0 / kPi, cc.close_width);
    }
  }

  // Model-based (Lecture 7): object dimensions are known — height = 40mm, x = 40mm.
  constexpr double kObjHeight = 0.040;
  const double grasp_z_nominal = obj_l0.point.z + 0.108;
  if (task1_debug_verbose_) {
    RCLCPP_INFO(
      node_->get_logger(),
      "[T1DBG] heights obj_z=%.4f obj_h=%.4f grasp_z_nominal(link8)=%.4f goal_z=%.4f",
      obj_l0.point.z, kObjHeight, grasp_z_nominal, goal_l0.point.z);
  }

  const double obj_h = kObjHeight;
  const std::string collision_id = "task1_object";

  const int max_tries = std::max(1, task1_max_candidates_to_try_);
  bool grasp_ok = false;
  Task1GraspCandidate used{};
  double last_grasp_cx = 0.0;
  double last_grasp_cy = 0.0;
  double last_grasp_z = 0.0;
  double last_lift_z = 0.0;

  for (int ti = 0; ti < std::min(max_tries, static_cast<int>(candidates.size())); ++ti) {
    const Task1GraspCandidate &c = candidates[static_cast<std::size_t>(ti)];
    const double yaw8 = task1_arm_link8_yaw(c.yaw);
    const double staging_yaw8 = task1_arm_link8_yaw(0.0);
    RCLCPP_INFO(
      node_->get_logger(), "Task1 try %d/%d: xy=(%.3f,%.3f) yaw=%.1fdeg close=%.4f score=%.1f", ti + 1,
      max_tries, c.x, c.y, c.yaw * 180.0 / kPi, c.close_width, c.score);

    // Open gripper BEFORE adding collision object — if gripper fingers are near the object
    // (e.g. after a previous failed try), the collision box would block hand planning.
    if (!set_gripper(task1_gripper_open_width_)) {
      RCLCPP_WARN(node_->get_logger(), "Task1: open gripper failed");
      continue;
    }

    const double grasp_z = grasp_z_nominal;
    const double grasp_z_final = grasp_z + std::max(0.0, task1_grasp_backoff_z_);
    // Known object top = obj_z + 40mm; hover well above it.
    const double obj_top_z = obj_l0.point.z + kObjHeight;
    const double pre_floor_from_roof = obj_top_z + 0.14;
    const double pre_z = std::max(
      {obj_l0.point.z + task1_pregrasp_offset_z_, grasp_z + std::max(0.06, task1_approach_last_delta_z_ + 0.05),
        pre_floor_from_roof});
    constexpr double k_hover_above_pre = 0.06;
    const double hover_z =
      std::min(0.48, std::max(pre_z + k_hover_above_pre, pre_floor_from_roof + 0.04));
    const double mid_z = std::min(hover_z - 0.012, std::max(pre_z + 0.028, grasp_z + 0.095));
    if (task1_debug_verbose_) {
      RCLCPP_INFO(
        node_->get_logger(),
        "[T1DBG] try %d hover_z=%.4f mid_z=%.4f pre_z=%.4f grasp_z=%.4f roof_floor=%.4f", ti + 1, hover_z, mid_z,
        pre_z, grasp_z_final, pre_floor_from_roof);
    }
    const geometry_msgs::msg::Pose pre_pose = make_topdown_pose(c.x, c.y, pre_z, yaw8);
    const geometry_msgs::msg::Pose obj_hover_stage =
      make_topdown_pose(obj_l0.point.x, obj_l0.point.y, hover_z, staging_yaw8);
    const geometry_msgs::msg::Pose cand_hover_stage =
      make_topdown_pose(c.x, c.y, hover_z, staging_yaw8);
    const geometry_msgs::msg::Pose cand_hover_aligned =
      make_topdown_pose(c.x, c.y, hover_z, yaw8);

    // Add collision object to protect against OMPL side-swipe during free-space moves.
    {
      moveit_msgs::msg::CollisionObject co;
      co.header.frame_id = "panda_link0";
      co.header.stamp = node_->get_clock()->now();
      co.id = collision_id;
      co.operation = moveit_msgs::msg::CollisionObject::ADD;
      shape_msgs::msg::SolidPrimitive box;
      box.type = shape_msgs::msg::SolidPrimitive::BOX;
      constexpr double kCollisionBoxH = 0.10;
      box.dimensions = {0.22, 0.22, kCollisionBoxH};
      geometry_msgs::msg::Pose box_pose;
      box_pose.position.x = obj_l0.point.x;
      box_pose.position.y = obj_l0.point.y;
      box_pose.position.z = obj_l0.point.z + kCollisionBoxH * 0.5;
      box_pose.orientation.w = 1.0;
      co.primitives.push_back(box);
      co.primitive_poses.push_back(box_pose);
      planning_scene_interface_.applyCollisionObject(co);
    }

    if (!move_arm_to_pose(obj_hover_stage)) {
      RCLCPP_WARN(node_->get_logger(), "Task1: move to object hover failed");
      planning_scene_interface_.removeCollisionObjects({collision_id});
      continue;
    }
    if (std::hypot(c.x - obj_l0.point.x, c.y - obj_l0.point.y) > 0.010 &&
        !move_arm_to_pose(cand_hover_stage))
    {
      RCLCPP_WARN(node_->get_logger(), "Task1: move to candidate hover failed");
      planning_scene_interface_.removeCollisionObjects({collision_id});
      continue;
    }
    if (!move_arm_to_pose(cand_hover_aligned)) {
      RCLCPP_WARN(node_->get_logger(), "Task1: in-place rotate above target failed");
      planning_scene_interface_.removeCollisionObjects({collision_id});
      continue;
    }

    // Remove collision object before vertical descent so Cartesian path is not blocked.
    planning_scene_interface_.removeCollisionObjects({collision_id});

    std::vector<geometry_msgs::msg::Pose> drop_to_pre;
    if (mid_z > pre_z + 0.012 && mid_z + 0.01 < hover_z) {
      drop_to_pre.push_back(make_topdown_pose(c.x, c.y, mid_z, yaw8));
    }
    drop_to_pre.push_back(pre_pose);
    if (!cartesian_follow_waypoints(drop_to_pre, 0.80, 0.11, 0.16, 0.001)) {
      RCLCPP_WARN(node_->get_logger(), "Task1: Cartesian drop to pre-grasp failed");
      continue;
    }

    const double z_above_grasp = grasp_z + task1_approach_last_delta_z_;
    std::vector<geometry_msgs::msg::Pose> descend_hi;
    descend_hi.push_back(make_topdown_pose(c.x, c.y, z_above_grasp, yaw8));
    descend_hi.push_back(make_topdown_pose(c.x, c.y, grasp_z + 0.012, yaw8));
    if (!cartesian_follow_waypoints(
        descend_hi, 0.80, task1_descend_velocity_scale_, task1_descend_acceleration_scale_, 0.001))
    {
      RCLCPP_WARN(node_->get_logger(), "Task1: descend to grasp (upper) failed");
      continue;
    }
    std::vector<geometry_msgs::msg::Pose> descend_lo;
    descend_lo.push_back(make_topdown_pose(c.x, c.y, grasp_z + 0.012, yaw8));
    descend_lo.push_back(make_topdown_pose(c.x, c.y, grasp_z_final, yaw8));
    if (!cartesian_follow_waypoints(descend_lo, 0.72, 0.035, 0.055, 0.0005)) {
      RCLCPP_WARN(node_->get_logger(), "Task1: descend to grasp (final mm) failed");
      continue;
    }

    if (!close_gripper_for_grasp(c.close_width)) {
      RCLCPP_WARN(node_->get_logger(), "Task1: close gripper failed");
      continue;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(std::max(0, task1_grasp_settle_ms_)));

    const double lift_z = std::max(obj_l0.point.z + task1_lift_offset_z_, grasp_z + 0.055);
    const geometry_msgs::msg::Pose lift_pose = make_topdown_pose(c.x, c.y, lift_z, yaw8);
    if (task1_debug_verbose_) {
      RCLCPP_INFO(node_->get_logger(), "[T1DBG] lift target z=%.4f (obj_z+lift=%.4f grasp+0.055=%.4f)", lift_z,
        obj_l0.point.z + task1_lift_offset_z_, grasp_z + 0.055);
    }
    if (!cartesian_move(lift_pose, 0.72)) {
      RCLCPP_WARN(node_->get_logger(), "Task1: lift after grasp failed");
      continue;
    }

    grasp_ok = true;
    used = c;
    last_grasp_cx = c.x;
    last_grasp_cy = c.y;
    last_grasp_z = grasp_z_final;
    last_lift_z = lift_z;
    break;
  }

  if (!grasp_ok) {
    RCLCPP_ERROR(node_->get_logger(), "Task1: all grasp attempts failed");
    return;
  }

  const double extra_release = std::clamp(
    task1_release_object_height_scale_ * kObjHeight, task1_release_min_extra_z_, task1_release_max_extra_z_);
  const double release_z = goal_l0.point.z + task1_place_offset_z_ + extra_release;

  const double safe_approach_z = std::max(
    last_lift_z + task1_transit_raise_above_lift_,
    std::max(
      goal_l0.point.z + task1_basket_approach_min_z_,
      std::max(last_grasp_z + 0.14, release_z + 0.06)));

  if (task1_debug_verbose_) {
    RCLCPP_INFO(
      node_->get_logger(),
      "[T1DBG] place-transit last_lift_z=%.4f last_grasp_z=%.4f goal_z=%.4f release_z=%.4f safe_approach_z=%.4f "
      "(need>=goal+%.3f lift+%.3f)",
      last_lift_z, last_grasp_z, goal_l0.point.z, release_z, safe_approach_z, task1_basket_approach_min_z_,
      task1_transit_raise_above_lift_);
  }

  const double yaw_place = task1_arm_link8_yaw(used.yaw);

  // Add ground plane collision to prevent any OMPL fallback from diving into the table.
  const std::string ground_id = "task1_ground_guard";
  {
    moveit_msgs::msg::CollisionObject gnd;
    gnd.header.frame_id = "panda_link0";
    gnd.header.stamp = node_->get_clock()->now();
    gnd.id = ground_id;
    gnd.operation = moveit_msgs::msg::CollisionObject::ADD;
    shape_msgs::msg::SolidPrimitive slab;
    slab.type = shape_msgs::msg::SolidPrimitive::BOX;
    slab.dimensions = {2.0, 2.0, 0.01};
    geometry_msgs::msg::Pose slab_pose;
    slab_pose.position.z = -0.005;
    slab_pose.orientation.w = 1.0;
    gnd.primitives.push_back(slab);
    gnd.primitive_poses.push_back(slab_pose);
    planning_scene_interface_.applyCollisionObject(gnd);
  }

  // Vertical rise with Cartesian path — pure vertical, no sudden direction changes.
  const geometry_msgs::msg::Pose rise_at_grasp =
    make_topdown_pose(last_grasp_cx, last_grasp_cy, safe_approach_z, yaw_place);
  if (!cartesian_move(rise_at_grasp, 0.72)) {
    if (!move_arm_to_pose(rise_at_grasp)) {
      RCLCPP_ERROR(node_->get_logger(), "Task1: rise to transit height at grasp xy failed");
      planning_scene_interface_.removeCollisionObjects({ground_id});
      return;
    }
  }

  // Grasp point is offset from object centroid. Compensate so the OBJECT centre
  // (not the grasp point) lands at the basket centre.
  const double grip_dx = last_grasp_cx - obj_l0.point.x;
  const double grip_dy = last_grasp_cy - obj_l0.point.y;
  const double place_x = goal_l0.point.x - grip_dx;
  const double place_y = goal_l0.point.y - grip_dy;

  const geometry_msgs::msg::Pose hover_above_basket =
    make_topdown_pose(place_x, place_y, safe_approach_z, yaw_place);
  std::vector<geometry_msgs::msg::Pose> to_basket;
  to_basket.push_back(hover_above_basket);
  if (!cartesian_follow_waypoints(
      to_basket, 0.55, task1_transit_xy_vel_scale_, task1_transit_xy_acc_scale_, 0.002))
  {
    RCLCPP_WARN(
      node_->get_logger(),
      "Task1: horizontal transit to basket (Cartesian) low fraction — trying move_arm_to_pose");
    if (!move_arm_to_pose(hover_above_basket)) {
      RCLCPP_ERROR(node_->get_logger(), "Task1: move to hover above basket center failed");
      planning_scene_interface_.removeCollisionObjects({ground_id});
      return;
    }
  }

  // Remove ground guard before descending to place.
  planning_scene_interface_.removeCollisionObjects({ground_id});

  const geometry_msgs::msg::Pose at_release =
    make_topdown_pose(place_x, place_y, release_z, yaw_place);
  std::vector<geometry_msgs::msg::Pose> vertical_place;
  vertical_place.push_back(at_release);
  if (!cartesian_follow_waypoints(
      vertical_place, 0.72, task1_place_descend_vel_scale_, task1_place_descend_acc_scale_, 0.001))
  {
    RCLCPP_WARN(node_->get_logger(), "Task1: vertical place Cartesian failed — trying cartesian_move");
    if (!cartesian_move(at_release, 0.72)) {
      RCLCPP_ERROR(node_->get_logger(), "Task1: vertical descend to place failed");
      return;
    }
  }

  if (!set_gripper(task1_gripper_open_width_)) {
    RCLCPP_WARN(node_->get_logger(), "Task1: open gripper for release failed");
  }

  const geometry_msgs::msg::Pose retreat_pose =
    make_topdown_pose(place_x, place_y, safe_approach_z, yaw_place);
  std::vector<geometry_msgs::msg::Pose> retreat_up;
  retreat_up.push_back(retreat_pose);
  if (!cartesian_follow_waypoints(
      retreat_up, 0.55, task1_place_descend_vel_scale_, task1_place_descend_acc_scale_, 0.001))
  {
    (void)move_arm_to_pose(retreat_pose);
  }

  // Ensure collision object is cleaned up even if it was not removed earlier.
  planning_scene_interface_.removeCollisionObjects({collision_id});

  RCLCPP_INFO(node_->get_logger(), "Task1 completed");
}

void cw2::t2_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task2Service::Response> response)
{
  (void)request;
  response->mystery_object_num = -1;

  std::string frame_id;
  std::size_t point_count = 0;
  std::uint64_t sequence = 0;
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    frame_id = g_input_pc_frame_id_;
    point_count = g_cloud_ptr ? g_cloud_ptr->size() : 0;
    sequence = g_cloud_sequence_;
  }

  RCLCPP_WARN(
    node_->get_logger(),
    "Task 2 is not implemented in cw2_team_8. Latest cloud: seq=%llu frame='%s' points=%zu",
    static_cast<unsigned long long>(sequence),
    frame_id.c_str(),
    point_count);
}

void cw2::t3_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  response->total_num_shapes = 0;
  response->num_most_common_shape = 0;
  response->most_common_shape_vector.clear();

  std::string frame_id;
  std::size_t point_count = 0;
  std::uint64_t sequence = 0;
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    frame_id = g_input_pc_frame_id_;
    point_count = g_cloud_ptr ? g_cloud_ptr->size() : 0;
    sequence = g_cloud_sequence_;
  }

  RCLCPP_WARN(
    node_->get_logger(),
    "Task 3 is not implemented in cw2_team_8. Latest cloud: seq=%llu frame='%s' points=%zu",
    static_cast<unsigned long long>(sequence),
    frame_id.c_str(),
    point_count);
}

props = gym.get_actor_dof_properties(env, actor_handle)
props["driveMode"].fill(gymapi.DOF_MODE_NONE)
props["stiffness"].fill(0.0)
props["damping"].fill(0.0)
gym.set_actor_dof_properties(env, actor_handle, props)


# configure the joints for effort control mode (once)
props = gym.get_actor_dof_properties(env, actor_handle)
props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
props["stiffness"].fill(0.0)
props["damping"].fill(0.0)
gym.set_actor_dof_properties(env, actor_handle, props)

# apply efforts (every frame)
efforts = np.full(num_dofs, 100.0).astype(np.float32)
gym.apply_actor_dof_efforts(env, actor_handle, efforts)

props = gym.get_actor_dof_properties(env, actor_handle)
props["driveMode"].fill(gymapi.DOF_MODE_POS)
props["stiffness"].fill(1000.0)
props["damping"].fill(200.0)
gym.set_actor_dof_properties(env, actor_handle, props)


targets = np.zeros(num_dofs).astype('f')
gym.set_actor_dof_position_targets(env, actor_handle, targets)



dof_props = gym.get_actor_dof_properties(envs, actor_handles)
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']
ranges = upper_limits - lower_limits

pos_targets = lower_limits + ranges * np.random.random(num_dofs).astype('f')
gym.set_actor_dof_position_targets(env, actor_handle, pos_targets)


dof_props = gym.get_actor_dof_properties(envs, actor_handles)
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']
ranges = upper_limits - lower_limits

pos_targets = lower_limits + ranges * np.random.random(num_dofs).astype('f')
gym.set_actor_dof_position_targets(env, actor_handle, pos_targets)

vel_targets = np.random.uniform(-math.pi, math.pi, num_dofs).astype('f')
gym.set_actor_dof_velocity_targets(env, actor_handle, vel_targets)
body_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_ALL)
body_states = gym.get_env_rigid_body_states(env, gymapi.STATE_ALL)
body_states = gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL)

body_states["pose"]             # all poses (position and orientation)
body_states["pose"]["p"])           # all positions (Vec3: x, y, z)
body_states["pose"]["r"])           # all orientations (Quat: x, y, z, w)
body_states["vel"]              # all velocities (linear and angular)
body_states["vel"]["linear"]    # all linear velocities (Vec3: x, y, z)
body_states["vel"]["angular"]   # all angular velocities (Vec3: x, y, z)

gym.set_actor_rigid_body_states(env, actor_handle, body_states, gymapi.STATE_ALL)
gym.set_env_rigid_body_states(env, body_states, gymapi.STATE_ALL)
gym.set_sim_rigid_body_states(sim, body_states, gymapi.STATE_ALL)

i1 = gym.find_actor_rigid_body_index(env, actor_handle, "body_name", gymapi.DOMAIN_ACTOR)
i2 = gym.find_actor_rigid_body_index(env, actor_handle, "body_name", gymapi.DOMAIN_ENV)
i3 = gym.find_actor_rigid_body_index(env, actor_handle, "body_name", gymapi.DOMAIN_SIM)

dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)

gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)


dof_states["pos"]   # all positions
dof_states["vel"]   # all velocities

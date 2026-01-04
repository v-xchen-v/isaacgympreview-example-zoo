# Ref: https://docs.robotsfan.com/isaacgym/programming/physics.html#actor-components


    
num_bodies = gym.get_actor_rigid_body_count(env, actor_handle)
num_joints = gym.get_actor_joint_count(env, actor_handle)
num_dofs = gym.get_actor_dof_count(env, actor_handle)
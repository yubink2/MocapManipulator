from inverse_kinematics import *
from h36m_dataset import Human36mDataset
from camera import *

import numpy as np
# In[2]:

joint_info = {
    'joint_name': [
        'root', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle',
        'chest', 'neck', 'nose', 'eye', 'left_shoulder', 'left_elbow', 'left_wrist',
        'right_shoulder', 'right_elbow', 'right_wrist'
    ],
    'father': [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
    'side': [
        'middle', 'right', 'right', 'right', 'left', 'left', 'left', 'middle', 'middle', 'middle',
        'middle', 'left', 'left', 'left', 'right', 'right', 'right'
    ]
}

# In[3]:


def init_fb_h36m_dataset(dataset_path):
  dataset = Human36mDataset(dataset_path)
  print('Preparing Facebook Human3.6M Dataset...')
  traj_total = 0
  for subject in dataset.subjects():
    for action in dataset[subject].keys():
      anim = dataset[subject][action]
      positions_3d = []
      for cam in anim['cameras']:
        pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
        pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
        positions_3d.append(pos_3d)
      
      anim['positions_3d'] = positions_3d
      ##############
      trajs = count_trajectories(positions_3d, cam)
      print(f'{subject} {action} has {trajs} trajectories')
      traj_total += trajs
      ##############
  return dataset

def count_trajectories(positions_3d, cam):
  pose_seq = positions_3d[0].copy()
  trajectory = pose_seq[:, :1]
  pose_seq[:, 1:] += trajectory
  # print("TEST positions_3d[0]: ", positions_3d[0])
  # print("TEST trajectory: ", trajectory)

  # Invert camera transformation
  pose_seq = camera_to_world(pose_seq, R=cam['orientation'], t=cam['translation'])
  # 10: head, 0: hip
  # 11: right shoulder, 12: right elbow, 13: right wrist
  # 14: left shoulder, 15: left elbow, 16: left wrist
  x = pose_seq[:, :, 0:1]
  x = x - np.expand_dims(np.repeat(x[:, 0, :], 17, axis=1), -1)
  px = x.copy(); px[:, 0:11, :] = x[0, 0:11, :]

  z = pose_seq[:, :, 1:2] * -1
  z = z - np.expand_dims(np.repeat(z[:, 0, :], 17, axis=1), -1)
  z[:, :, 0] += 0.5
  pz = z.copy(); pz[:, 0:11, :] = z[0, 0:11, :]

  y = pose_seq[:, :, 2:3]
  y = y - np.expand_dims(np.repeat(y[:, 0, :], 17, axis=1), -1)
  py = y.copy(); py[:, 0:11, :] = y[0, 0:11, :]

  pose_seq = np.concatenate((px, pz, py), axis=2)
  filtered_idx = np.sum(pose_seq[:, 11:, 1] >= 0.45, axis=1)
  filtered_idx = np.where(filtered_idx == 6)
  if len(filtered_idx) == 0 or len(filtered_idx[0]) == 0:
    return 0
  else:
    filtered_idx = filtered_idx[0]
  
  last = filtered_idx[0]
  count, seg_len = 0, 10
  trajs = 0
  for i in filtered_idx:
    if i - last <= 1:
      count += 1
    else:
      if count > seg_len:
        trajs += 1
      count = 0
    last = i

  return trajs


def pose3D_from_fb_h36m(dataset, subject, action, shift):
  # pose_seq = dataset[subject][action]['positions_3d'][0].copy()
  # # print("TEST shape of pose_seq: ", pose_seq.shape)  # shape: (1621, 17, 3)
  # print("TEST pose_seq[0][11]: ", pose_seq[0][11])

  # trajectory = pose_seq[:, :1]
  # pose_seq[:, 1:] += trajectory
  # # Invert camera transformation
  # cam = dataset.cameras()[subject][0]
  # pose_seq = camera_to_world(pose_seq, R=cam['orientation'], t=cam['translation'])

  # ##################################################################################
  # # 10: head, 0: hip
  # # 11: right shoulder, 12: right elbow, 13: right wrist
  # # 14: left shoulder, 15: left elbow, 16: left wrist

  # #### eliminating floating around actions by copying one frame?
  # x = pose_seq[:, :, 0:1]
  # x = x - np.expand_dims(np.repeat(x[:, 0, :], 17, axis=1), -1)
  # px = x.copy(); px[:, 4:, :] = x[0, 4:, :]

  # z = pose_seq[:, :, 1:2] * -1
  # z = z - np.expand_dims(np.repeat(z[:, 0, :], 17, axis=1), -1)
  # z[:, :, 0] += 0.5
  # pz = z.copy(); #pz[:, 0:11, :][pz[:, 0:11, :] < 0.5] = 0.5#z[0, 0:11, :]
  # pz[:, 4:, :] = z[0, 4:, :]
  # # pz[:, 6, :] += 0.05; pz[:, 3, :] += 0.05
  # # pz[:, 3:5, :] += 0.05; pz[:, 6:8, :] += 0.05

  # y = pose_seq[:, :, 2:3]
  # y = y - np.expand_dims(np.repeat(y[:, 0, :], 17, axis=1), -1)
  # py = y.copy(); py[:, 4:, :] = y[0, 4:, :]


  # pose_seq = np.concatenate((px, pz, py), axis=2)

  # #### ????? wat is dis........
  # filtered_idx = np.sum(pose_seq[:, :4, 1] >= 0.5, axis=1)
  # filtered_idx = (filtered_idx == 4)
  # print("TEST filtered_idx: ", filtered_idx)
  # pose_seq = pose_seq[filtered_idx]
  # ##################################################################################

  # # plus shift
  # pose_seq += np.array([[shift for i in range(pose_seq.shape[1])] for j in range(pose_seq.shape[0])])

  # # print("TEST2 shape of pose_seq: ", pose_seq.shape)  # shape: (482, 17, 3)
  # print("TEST2 pose_seq[0][11]: ", pose_seq[0][11])
  # return pose_seq

  print('pose3D_from_fb_h36m()')

  pose_seq = dataset[subject][action]['positions_3d'][0].copy()
  trajectory = pose_seq[:, :1]
  pose_seq[:, 1:] += trajectory
  # Invert camera transformation
  cam = dataset.cameras()[subject][0]
  pose_seq = camera_to_world(pose_seq, R=cam['orientation'], t=cam['translation'])
  x = pose_seq[:, :, 0:1]
  y = pose_seq[:, :, 1:2]
  z = pose_seq[:, :, 2:3]

  pose_seq = np.concatenate((x, z, y), axis=2)

  # plus shift
  pose_seq += np.array([[shift for i in range(pose_seq.shape[1])] for j in range(pose_seq.shape[0])
                       ])
  return pose_seq


def rot_seq_to_deepmimic_json(rot_seq, loop, json_path):
  to_json = {"Loop": loop, "Frames": []}
  rot_seq = np.around(rot_seq, decimals=6)
  to_json["Frames"] = rot_seq.tolist()
  # In[14]:
  to_file = json.dumps(to_json)
  file = open(json_path, "w")
  file.write(to_file)
  file.close()

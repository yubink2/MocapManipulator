with open(control_points_json, "rt") as json_file:
                    control_points = json.load(json_file)

                # Write control point locations in link frames as transforms
                self.control_points = dict()
                for link_name, ctrl_point_list in control_points.items():
                    # print('*link_name: ', link_name)
                    # print('*ctlr_point_list: ', ctrl_point_list)
                    self.control_points[link_name] = []
                    for ctrl_point in ctrl_point_list:
                        ctrl_pose_link_frame = torch.eye(4, device = self._device, dtype = self._float_dtype)
                        ctrl_pose_link_frame[:3,3] = torch.tensor(ctrl_point, device = self._device, dtype = self._float_dtype)
                        self.control_points[link_name].append(ctrl_pose_link_frame)
                    self.control_points[link_name] = torch.stack(self.control_points[link_name])
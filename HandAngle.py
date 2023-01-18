import torch

def HandAngle(landmark):
    N = 21 # number of landmark points and connection points
    angle = torch.zeros(int(N*(N-1)/2)) # total number of angles from N points
    
    landmark_points = landmark.view(N, 3)
    hand_connections = [(3, 4), (0, 5), (17, 18), (0, 17),
                        (13, 14), (13, 17), (18, 19), (5, 6),
                        (5, 9), (14, 15), (0, 1), (9, 10),
                        (1, 2), (9, 13), (10, 11), (19, 20),
                        (6, 7), (15, 16), (2, 3), (11, 12), (7, 8)] # connection points
    
    count = 0
    for i in range(N):
        for j in range(N):
            if i < j:
                # get 2 vectors from 3 landmark points
                u = landmark_points[hand_connections[i][0], :] - landmark_points[hand_connections[i][1]]
                v = landmark_points[hand_connections[j][0], :] - landmark_points[hand_connections[j][1]]
                u = u.view(-1)
                v = v.view(-1)
                # calculate dot product, norm, angle
                dot_product = torch.dot(u, v)
                norm = torch.norm(u) * torch.norm(v)
                angle[count] = torch.arccos(dot_product/norm)
                count += 1
    
    return angle

if __name__ == "__main__":
    print("HandAngle module is running.")
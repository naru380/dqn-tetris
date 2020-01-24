from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt



resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])
screen_width = 600
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def get_cart_location():
#     world_width = env.x_threshold * 2
#     scale = screen_width / world_width
#     return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    # screen = screen[:, 160:320]
    # view_width = 320
    # cart_location = get_cart_location()
    # if cart_location < view_width // 2:
    #     slice_range = slice(view_width)
    # elif cart_location > (screen_width - view_width // 2):
    #     slice_range = slice(-view_width, None)
    # else:
    #     slice_range = slice(cart_location - view_width // 2,
    #                         cart_location + view_width // 2)
    # # Strip off the edges, so that we have a square image centered on a cart
    # screen = screen[:, :, slice_range]
    # # Convert to float, rescare, convert to torch tensor
    # # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


if __name__ == '__main__':
    # import gym
    # env = gym.make('CartPole-v0').unwrapped
    import gym_tetris
    env = gym_tetris.make('TetrisA-v0')
    env.reset()
    plt.figure()
    plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
    plt.title('Example extracted screen')
    plt.show()

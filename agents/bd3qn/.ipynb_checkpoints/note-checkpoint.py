{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 222,
   "id": "refined-canberra",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "import numpy as np\n",
    "from noisy_net import NoisyLinear, NoisyFactorizedLinear\n",
    "from OneHotEncode import OneHotEncode\n",
    "from config import Configuration\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 244,
   "id": "gorgeous-shakespeare",
   "metadata": {},
   "outputs": [],
   "source": [
    "class BranchingQNetwork(nn.Module):\n",
    "    def __init__(self, observation_space, action_bins, hidden_dim, device, exploration_method=\"Epsilon\"):\n",
    "        super().__init__()\n",
    "        self.exploration_method = exploration_method\n",
    "        self.device = device\n",
    "        self.model = nn.ModuleList([nn.Sequential(\n",
    "            nn.Linear(observation_space, hidden_dim*4),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(hidden_dim*4, hidden_dim*2),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(hidden_dim*2, hidden_dim),\n",
    "            nn.ReLU()\n",
    "        ) for i in range(12)])\n",
    "        self.value_head = nn.ModuleList([nn.Linear(hidden_dim, 1) for i in range(12)])\n",
    "        self.adv_heads = nn.ModuleList(nn.Linear(hidden_dim, 11) for i in range(12))\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.state_pre_processing(x)\n",
    "        print(x)\n",
    "        \n",
    "        out = [[o(t) for t in x] for o in self.model]\n",
    "        value = [v(out) for v in self.value_head]\n",
    "#         if value.shape[0] == 1:\n",
    "#             advs = torch.stack([l(out) for l in self.adv_heads], dim=0)\n",
    "#             q_val = value + advs - advs.mean(1, keepdim=True)\n",
    "#         else:\n",
    "#             advs = torch.stack([l(out) for l in self.adv_heads], dim=1)\n",
    "#             q_val = value.unsqueeze(1) + advs - advs.mean(2, keepdim=True)\n",
    "        return value\n",
    "\n",
    "    def sample_noise(self):\n",
    "        self.model[0].sample_noise()\n",
    "        self.model[2].sample_noise()\n",
    "        self.model[4].sample_noise()\n",
    "        self.value_head.sample_noise()\n",
    "        for l in self.adv_heads:\n",
    "            l.sample_noise()\n",
    "            "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "pressed-wichita",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 297,
   "id": "second-jacob",
   "metadata": {},
   "outputs": [],
   "source": [
    "obs = [  64.,    0.,    0.,  500.,    0.,    0.,    1.,  100.,    8.,\n",
    "          0.,    0.,  100.,    0.,    1.,    0.,  100.,    0.,    0.,\n",
    "          0.,  -88.,    0.,    0.,    0., -100.,   16.,    0.,    0.,\n",
    "        100.,   12.,    0.,    1., -100.,   16.,    0.,    0.,  -50.,\n",
    "          7.,    1.,    0.,  -13.,    8.,    0.,    0., -500.,   32.,\n",
    "          3.,    1.,   94.,    1.,    8.,   10.,    2.,   56.,    1.,\n",
    "          7.,    8.,    0.,   93.,    0.,    8.,    6.,    1.,   58.,\n",
    "          0.,    8.,    8.,    2.,    0.,    0.,    0.,    4.,    0.,\n",
    "        100.,    0.,    8.,    7.,    1.,   90.,    0.,    8.,    2.,\n",
    "          2.,  100.,    0.,    8.,    7.,    0.,  100.,    0.,    8.,\n",
    "          2.,    1.,   15.,    0.,    7.,    9.,    2.,   70.,    0.,\n",
    "          8.,    2.,    0.,   91.,    1.,   12.]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 298,
   "id": "cleared-attribute",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "config = Configuration('configs/config.json')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 299,
   "id": "victorian-album",
   "metadata": {},
   "outputs": [],
   "source": [
    "torch.cuda.init()\n",
    "device = torch.device(\n",
    "    config.device if torch.cuda.is_available() else \"cpu\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 300,
   "id": "prompt-oxide",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = BranchingQNetwork(62, 11, 128, device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 301,
   "id": "confirmed-placement",
   "metadata": {},
   "outputs": [],
   "source": [
    "def state_pre_processing(obs):\n",
    "    new_obs = OneHotEncode(obs)\n",
    "    node_info = new_obs[:45]\n",
    "    groups_info = new_obs[45:]\n",
    "    split_groups_info = []\n",
    "    i = 0\n",
    "    while i < len(groups_info):\n",
    "        #print(\"nodes\", node_info)\n",
    "        #print(\"groups\", groups_info[i:i+17])\n",
    "        group = np.concatenate((node_info, groups_info[i:i+17]))\n",
    "        split_groups_info.append(group)\n",
    "        i = i + 17\n",
    "\n",
    "    return split_groups_info"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 304,
   "id": "referenced-programming",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  64,    0,    0,  500,    0,    0,    1,  100,    8,    0,    0,\n",
       "         100,    0,    1,    0,  100,    0,    0,    0,  -88,    0,    0,\n",
       "           0, -100,   16,    0,    0,  100,   12,    0,    1, -100,   16,\n",
       "           0,    0,  -50,    7,    1,    0,  -13,    8,    0,    0, -500,\n",
       "          32,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,\n",
       "           0,    0,    1,    0,   94,    1,    8],\n",
       "       [  64,    0,    0,  500,    0,    0,    1,  100,    8,    0,    0,\n",
       "         100,    0,    1,    0,  100,    0,    0,    0,  -88,    0,    0,\n",
       "           0, -100,   16,    0,    0,  100,   12,    0,    1, -100,   16,\n",
       "           0,    0,  -50,    7,    1,    0,  -13,    8,    0,    0, -500,\n",
       "          32,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
       "           1,    0,    0,    1,   56,    1,    7],\n",
       "       [  64,    0,    0,  500,    0,    0,    1,  100,    8,    0,    0,\n",
       "         100,    0,    1,    0,  100,    0,    0,    0,  -88,    0,    0,\n",
       "           0, -100,   16,    0,    0,  100,   12,    0,    1, -100,   16,\n",
       "           0,    0,  -50,    7,    1,    0,  -13,    8,    0,    0, -500,\n",
       "          32,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0,\n",
       "           0,    1,    0,    0,   93,    0,    8],\n",
       "       [  64,    0,    0,  500,    0,    0,    1,  100,    8,    0,    0,\n",
       "         100,    0,    1,    0,  100,    0,    0,    0,  -88,    0,    0,\n",
       "           0, -100,   16,    0,    0,  100,   12,    0,    1, -100,   16,\n",
       "           0,    0,  -50,    7,    1,    0,  -13,    8,    0,    0, -500,\n",
       "          32,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,\n",
       "           0,    0,    1,    0,   58,    0,    8],\n",
       "       [  64,    0,    0,  500,    0,    0,    1,  100,    8,    0,    0,\n",
       "         100,    0,    1,    0,  100,    0,    0,    0,  -88,    0,    0,\n",
       "           0, -100,   16,    0,    0,  100,   12,    0,    1, -100,   16,\n",
       "           0,    0,  -50,    7,    1,    0,  -13,    8,    0,    0, -500,\n",
       "          32,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0,\n",
       "           0,    0,    0,    1,    0,    0,    0],\n",
       "       [  64,    0,    0,  500,    0,    0,    1,  100,    8,    0,    0,\n",
       "         100,    0,    1,    0,  100,    0,    0,    0,  -88,    0,    0,\n",
       "           0, -100,   16,    0,    0,  100,   12,    0,    1, -100,   16,\n",
       "           0,    0,  -50,    7,    1,    0,  -13,    8,    0,    0, -500,\n",
       "          32,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,\n",
       "           0,    1,    0,    0,  100,    0,    8],\n",
       "       [  64,    0,    0,  500,    0,    0,    1,  100,    8,    0,    0,\n",
       "         100,    0,    1,    0,  100,    0,    0,    0,  -88,    0,    0,\n",
       "           0, -100,   16,    0,    0,  100,   12,    0,    1, -100,   16,\n",
       "           0,    0,  -50,    7,    1,    0,  -13,    8,    0,    0, -500,\n",
       "          32,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,\n",
       "           0,    0,    1,    0,   90,    0,    8],\n",
       "       [  64,    0,    0,  500,    0,    0,    1,  100,    8,    0,    0,\n",
       "         100,    0,    1,    0,  100,    0,    0,    0,  -88,    0,    0,\n",
       "           0, -100,   16,    0,    0,  100,   12,    0,    1, -100,   16,\n",
       "           0,    0,  -50,    7,    1,    0,  -13,    8,    0,    0, -500,\n",
       "          32,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,\n",
       "           0,    0,    0,    1,  100,    0,    8],\n",
       "       [  64,    0,    0,  500,    0,    0,    1,  100,    8,    0,    0,\n",
       "         100,    0,    1,    0,  100,    0,    0,    0,  -88,    0,    0,\n",
       "           0, -100,   16,    0,    0,  100,   12,    0,    1, -100,   16,\n",
       "           0,    0,  -50,    7,    1,    0,  -13,    8,    0,    0, -500,\n",
       "          32,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,\n",
       "           0,    1,    0,    0,  100,    0,    8],\n",
       "       [  64,    0,    0,  500,    0,    0,    1,  100,    8,    0,    0,\n",
       "         100,    0,    1,    0,  100,    0,    0,    0,  -88,    0,    0,\n",
       "           0, -100,   16,    0,    0,  100,   12,    0,    1, -100,   16,\n",
       "           0,    0,  -50,    7,    1,    0,  -13,    8,    0,    0, -500,\n",
       "          32,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,\n",
       "           0,    0,    1,    0,   15,    0,    7],\n",
       "       [  64,    0,    0,  500,    0,    0,    1,  100,    8,    0,    0,\n",
       "         100,    0,    1,    0,  100,    0,    0,    0,  -88,    0,    0,\n",
       "           0, -100,   16,    0,    0,  100,   12,    0,    1, -100,   16,\n",
       "           0,    0,  -50,    7,    1,    0,  -13,    8,    0,    0, -500,\n",
       "          32,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,\n",
       "           0,    0,    0,    1,   70,    0,    8],\n",
       "       [  64,    0,    0,  500,    0,    0,    1,  100,    8,    0,    0,\n",
       "         100,    0,    1,    0,  100,    0,    0,    0,  -88,    0,    0,\n",
       "           0, -100,   16,    0,    0,  100,   12,    0,    1, -100,   16,\n",
       "           0,    0,  -50,    7,    1,    0,  -13,    8,    0,    0, -500,\n",
       "          32,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,\n",
       "           0,    1,    0,    0,   91,    1,   12]])"
      ]
     },
     "execution_count": 304,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_obs = state_pre_processing(obs)\n",
    "new_obs = np.array(new_obs)\n",
    "new_obs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 296,
   "id": "young-dating",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "expected np.ndarray (got list)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-296-dea5d1c210cd>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mobs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfrom_numpy\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mobs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfloat\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mobs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mobs\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mto\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdevice\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0mobs\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: expected np.ndarray (got list)"
     ]
    }
   ],
   "source": [
    "obs = torch.from_numpy(obs).float()\n",
    "obs = obs.to(device)\n",
    "obs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 250,
   "id": "tested-drilling",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "state_pre_processing() takes 1 positional argument but 2 were given",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-250-28b2254d7767>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mmodel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mobs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32mD:\\ProgramData\\Anaconda3\\envs\\everglades\\lib\\site-packages\\torch\\nn\\modules\\module.py\u001b[0m in \u001b[0;36m_call_impl\u001b[1;34m(self, *input, **kwargs)\u001b[0m\n\u001b[0;32m    887\u001b[0m             \u001b[0mresult\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_slow_forward\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0minput\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    888\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 889\u001b[1;33m             \u001b[0mresult\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mforward\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0minput\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    890\u001b[0m         for hook in itertools.chain(\n\u001b[0;32m    891\u001b[0m                 \u001b[0m_global_forward_hooks\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m<ipython-input-244-3d311a538fa9>\u001b[0m in \u001b[0;36mforward\u001b[1;34m(self, x)\u001b[0m\n\u001b[0;32m     16\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     17\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mforward\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 18\u001b[1;33m         \u001b[0mx\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstate_pre_processing\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     19\u001b[0m         \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     20\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: state_pre_processing() takes 1 positional argument but 2 were given"
     ]
    }
   ],
   "source": [
    "model(obs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "southwest-exception",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aggregate-rates",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "governing-credit",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

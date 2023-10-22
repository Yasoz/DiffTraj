#### Requirements
```bash
python>=3.7<br>
torch>=1.7<br>
pandas<br>
numpy<br>
matplotlib<br>
pathlib<br>
shutil<br>
datetime<br>
colored<br>
math<br>
```

#### Train the trajectory generation model

1. Download the data with permission of using your own GPS trajectory data

2. Change the corresponding paths in the config.py and main.py files

3. Run and train the model by:

   ```bash
   python main.py 
   ```

4. The code will save the trained model every 10 epoches

#### Usage for Trajectory Generation

Refer to Traj_Generation.ipynb file, we provide the fine-trained model (model.pt) simple guide infromation(heads.npy) example to explain how to generate trajectories.

#### Generated result

![Chengdu](./Chengdu_traj.png)

#### Requirements

python>=3.7
torch>=1.7
pandas
numpy
matplotlib
pathlib
shutil
datetime
colored
math

#### Train the trajectory generation model

1. Processing of trajectory data into fixed length vectors (refer to Trajdata_processing.ipynb)

2. Change the corresponding paths in the config.py and main.py files

3. Run and train the model by:

   ```bash
   python main.py 
   ```

4. The code will save the trained model every 10 epoches

#### Usage for Trajectory Generation

Refer to Traj_Generation.ipynb file

#### Generated result

![Chengdu](./Chengdu_traj.png)

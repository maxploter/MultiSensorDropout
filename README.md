# Multi sensor dropout

### Compiling CUDA operators
```bash
module avail cuda
module load cuda/12.1.0

cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

# 24-Hour Training Guide

This guide explains how to use the 24-hour continuous training feature to train a high-quality TSP parameter prediction model.

## Overview

The 24-hour training mode (`train_24h.py`) trains the neural network on randomly generated TSP instances over an extended period. Unlike the quick training mode in `main.py`, this approach:

- Generates diverse TSP instances with varying sizes (10-550 cities)
- Evaluates multiple parameter configurations for each instance
- Finds optimal Simulated Annealing parameters empirically
- Continuously accumulates training data and improves the model
- Saves periodic checkpoints to prevent data loss

## Basic Usage

### Full 24-Hour Training

To train for 24 hours with default settings:

```bash
python train_24h.py
```

This will:
- Train for 24 hours
- Generate instances with 10-550 cities
- Test 15 parameter configurations per instance
- Save checkpoints every 60 minutes
- Save the final model to `parameter_model_24h.pth`

### Custom Duration

For shorter or longer training:

```bash
# Train for 12 hours
python train_24h.py --duration 12.0

# Train for 48 hours
python train_24h.py --duration 48.0

# Quick test (6 minutes)
python train_24h.py --duration 0.1
```

## Advanced Configuration

### City Range

Customize the range of problem sizes:

```bash
# Focus on small to medium problems (10-100 cities)
python train_24h.py --min-cities 10 --max-cities 100

# Focus on large problems (100-1000 cities)
python train_24h.py --min-cities 100 --max-cities 1000
```

### Training Parameters

Fine-tune the training process:

```bash
python train_24h.py \
    --duration 24.0 \
    --batch-size 100 \
    --epochs 50 \
    --learning-rate 0.0005 \
    --param-configs 20
```

Parameters:
- `--batch-size`: Number of instances before retraining (default: 50)
- `--epochs`: Training epochs per batch (default: 20)
- `--learning-rate`: Learning rate (default: 0.001)
- `--param-configs`: Number of SA parameter sets to test per instance (default: 15)

### Checkpoint Interval

Control how often the model is saved:

```bash
# Save every 30 minutes
python train_24h.py --checkpoint-interval 30

# Save every 2 hours (more efficient for very long runs)
python train_24h.py --checkpoint-interval 120
```

### Custom Model Path

Save the model to a specific location:

```bash
python train_24h.py --model-path /path/to/my_model.pth
```

## Resuming Training

If training is interrupted, simply run the command again. The script will:
1. Load the existing model from `parameter_model_24h.pth`
2. Load accumulated training data from `training_features_24h.npy` and `training_params_24h.npy`
3. Continue training from where it left off

Example:
```bash
# Start training
python train_24h.py --duration 24.0

# If interrupted, resume by running the same command
python train_24h.py --duration 24.0
```

## Using the Trained Model

After training, use the model for predictions:

```python
from neural_network import ParameterPredictor
from tsp_solver import generate_random_tsp

# Load the trained model
predictor = ParameterPredictor(model_path='parameter_model_24h.pth')
predictor.load()

# Generate a TSP instance
tsp = generate_random_tsp(n_cities=100)
features = tsp.get_features()

# Predict optimal parameters
params = predictor.predict(features)
initial_temp, cooling_rate, min_temp, iterations = params

print(f"Predicted parameters for 100-city TSP:")
print(f"  Initial temperature: {initial_temp:.2f}")
print(f"  Cooling rate: {cooling_rate:.4f}")
print(f"  Min temperature: {min_temp:.4f}")
print(f"  Iterations per temp: {int(iterations)}")
```

Or use the existing solve script:

```bash
# Update main.py to use parameter_model_24h.pth instead of parameter_model.pth
# Then solve a TSP instance
python main.py --solve --cities 200
```

## Example Training Sessions

### Quick Quality Model (2-4 hours)
Good for testing and moderate-quality results:

```bash
python train_24h.py \
    --duration 3.0 \
    --max-cities 200 \
    --batch-size 30 \
    --epochs 15
```

### Production Quality (24 hours)
Recommended for production use:

```bash
python train_24h.py \
    --duration 24.0 \
    --max-cities 550 \
    --batch-size 50 \
    --epochs 20 \
    --param-configs 15
```

### Extended Training (72 hours)
For maximum quality on very large problems:

```bash
python train_24h.py \
    --duration 72.0 \
    --min-cities 10 \
    --max-cities 1000 \
    --batch-size 75 \
    --epochs 30 \
    --param-configs 20 \
    --learning-rate 0.0003
```

## Monitoring Progress

During training, the script displays:
- Current progress percentage
- Number of instances processed
- Elapsed and remaining time
- Size of the last processed instance

Example output:
```
[42.3%] Processed 245 instances | Elapsed: 10.15h | Remaining: 13.85h | Last instance: 342 cities
```

Checkpoints show:
- Total training samples accumulated
- Model save location
- Progress statistics

## Tips for Best Results

1. **Duration**: Longer training generally produces better models, but returns diminish after 24-48 hours

2. **City Range**: 
   - Include the full range (10-550) for general-purpose models
   - Focus on specific ranges if you know your problem sizes

3. **Parameter Configs**: 
   - More configs (20-30) find better parameters but slow down training
   - 15 configs is a good balance

4. **Batch Size**:
   - Larger batches (50-100) provide more stable training
   - Smaller batches (20-30) update the model more frequently

5. **Learning Rate**:
   - Start with default (0.001)
   - Reduce (0.0005-0.0003) for longer training sessions
   - Increase (0.002-0.005) for short training sessions

6. **Hardware**: 
   - The script automatically uses GPU if available
   - Training is CPU-intensive during parameter evaluation
   - More CPU cores = faster instance processing

## Troubleshooting

**Q: Training is very slow**
- Reduce `--param-configs` to 10 or less
- Reduce `--max-cities` to focus on smaller problems
- Increase `--batch-size` to train less frequently

**Q: Model quality is poor**
- Train for longer (24+ hours)
- Increase `--param-configs` to 20-30
- Ensure full city range (10-550)

**Q: Out of memory**
- Reduce `--batch-size`
- Reduce `--max-cities`
- Use smaller `--epochs`

**Q: How to stop training?**
- Press Ctrl+C to stop gracefully
- The final model will be saved
- You can resume later

## Files Generated

During and after training, these files are created:

- `parameter_model_24h.pth`: The trained neural network model
- `training_features_24h.npy`: Accumulated feature vectors
- `training_params_24h.npy`: Accumulated optimal parameters

These files are automatically excluded from git (see `.gitignore`).

## Comparison with Quick Training

| Feature | Quick Training (`main.py --train`) | 24-Hour Training (`train_24h.py`) |
|---------|-----------------------------------|-----------------------------------|
| Duration | Minutes | Hours to days |
| Instances | 100-200 | Hundreds to thousands |
| Parameter Search | Heuristic-based | Empirical evaluation |
| Quality | Moderate | High |
| Use Case | Testing, demos | Production |

## Next Steps

After training:
1. Test the model on various problem sizes
2. Compare with default parameters using `main.py --compare`
3. Evaluate on TSPLIB benchmarks with `benchmark_comparison.py`
4. Use in your TSP applications

For more information, see the main README.md.

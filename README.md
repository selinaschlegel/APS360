# APS360

- Progress Report 5%: Friday, Nov 7 at 11:59 PM
- Final Report 10%: Wednesday, Dec 3 at 11:59 PM

### Usage

#### Running the Full Pipeline

```bash
# Train with specific sample count (e.g., 5000 samples)
python src/main.py --samples 5000 --trials 20

# Train with all available samples
python src/main.py --trials 20

# Train with combined training+validation data (for best performance)
python src/main.py --combined-data --trials 20
```

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate dummy employee data
np.random.seed(42)

# Settings
num_employees = 10
days = 60  # Number of days of records
start_date = datetime(2023, 1, 1)

data = []

for emp_id in range(1, num_employees + 1):
    for day in range(days):
        date = start_date + timedelta(days=day)
        tasks_completed = np.random.poisson(lam=10)  # average of 10 tasks/day
        work_hours = np.random.normal(loc=8, scale=1)  # average 8 hours/day
        performance_rating = np.clip(np.random.normal(loc=4, scale=0.5), 1, 5)  # scale 1 to 5
        data.append([emp_id, date.strftime('%Y-%m-%d'), int(tasks_completed), round(work_hours, 1), round(performance_rating, 1)])

df = pd.DataFrame(data, columns=['employee_id', 'date', 'tasks_completed', 'work_hours', 'performance_rating'])

# Save to CSV
df.to_csv('employee_data.csv', index=False)
print("Dummy employee_data.csv generated!")

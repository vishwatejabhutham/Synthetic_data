from sdv.datasets.demo import download_demo
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality
import pandas as pd 

print("Loading demo data")

real_data, metadata = download_demo(
    modality='single_table',
    dataset_name='fake_hotel_guests'
)
# Create new metadata for the subset

metadata_subset = SingleTableMetadata()
metadata_subset.detect_from_dataframe(real_data)

real_data.to_csv('real_data.csv', index=False)

print(f"Number of rows and columns in original data: {real_data.shape}")
print(f"Selected columns: {list(real_data.columns)}")

# Create and fit synthesizer
synthesizer = GaussianCopulaSynthesizer(metadata_subset)
synthesizer.fit(real_data)

# Generate 100000 synthetic rows
synthetic_data = synthesizer.sample(num_rows=100000)

# Evaluate the synthetic data
diagnostic = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

quality_report = evaluate_quality(
    real_data,
    synthetic_data,
    metadata
)

# Save models and data
synthesizer.save('my_synthesizer.pkl')
synthetic_data.to_csv('synthetic_data.csv', index=False)

print("Process completed successfully!")

print(f"Real data: {real_data.shape[0]} rows, {real_data.shape[1]} columns")

print(f"Synthetic data: {synthetic_data.shape[0]} rows, {synthetic_data.shape[1]} columns")


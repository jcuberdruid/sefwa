import pandas as pd
import matplotlib.pyplot as plt

def plot_fft_column(csv_file, column_index):
    # Load the specified column from the CSV file
    data = pd.read_csv(csv_file, usecols=[column_index])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(f'FFT Data - Column {column_index}')
    plt.xlabel('Sample')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_fft_column("fft_segment_4.csv", 29)  # Column index 28 for the 29th column (0-indexed)


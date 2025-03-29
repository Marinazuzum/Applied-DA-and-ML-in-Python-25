import numpy as np
import matplotlib.pyplot as plt

# Global variable for location names
LOC_NAME = 'Augsburg' #['Augsburg', 'Straubing', 'Zugspitze']

# Function to read data from a text file for a given location
def read_text(location):
    # Assuming the file is named based on the location
    file_path = f"1. Introduction to Python/06_Climate_Diagram_Generator/DataBase_{location}.txt"  # Update this to the correct file path
    
    date = []
    T_avg = []
    sun_time = []
    rainfall = []
    
    # Read the file line by line
    with open(file_path, 'r') as file:
        # Skip the header
        lines = file.readlines()[3:]  # Skip the first 5 lines (header)
        
        for line in lines:
            parts = line.strip().split()  # Split line into parts
            
            # Extract relevant columns (JJJJMMDD, TM, SO, RR)
            date.append(parts[1])  # Date in YYYYMMDD format
            T_avg.append(float(parts[5]))  # Temperature (TM) in 2 m height above ground
            sun_time.append(float(parts[10]))  # Sunshine duration (SO) in hours
            rainfall.append(float(parts[12]))  # Rainfall (RR) in mm
    
    # Convert data to numpy arrays
    T_avg = np.array(T_avg)
    sun_time = np.array(sun_time)
    rainfall = np.array(rainfall)
    
    # Return a tuple with the processed data
    return date, T_avg, sun_time, rainfall

# Function to evaluate sun data
def evaluate_sun_data(data, location):
    date, T_avg, sun_time, rainfall = data
    
    total_sun_time = np.sum(sun_time)
    avg_sun_time = np.mean(sun_time)
    max_sun_time = np.max(sun_time)
    max_sun_day = date[np.argmax(sun_time)]  # Get the date of the sunniest day
    
    print(f"In total there were {total_sun_time:.1f} hours of sun in {location}, which on average were {avg_sun_time:.2f} hours per day.")
    print(f"The sunniest day was on {max_sun_day[:4]}/{max_sun_day[4:6]}/{max_sun_day[6:]} with {max_sun_time:.1f} hours of sun.")

# Function to calculate monthly averages of temperature and accumulated rainfall
def calculate_month(data):
    date, T_avg, sun_time, rainfall = data
    
    months = np.array([int(d[4:6]) for d in date])  # Extract month from date (MM)
    
    # Calculate monthly averages and sum of rainfall
    mean_temp = np.zeros(12)
    accumulated_rainfall = np.zeros(12)
    
    for month in range(1, 13):
        month_mask = months == month
        mean_temp[month - 1] = np.mean(T_avg[month_mask])
        accumulated_rainfall[month - 1] = np.sum(rainfall[month_mask])
    
    return mean_temp, accumulated_rainfall

# Function to plot a climate diagram
def plot_climate_diagram(temp_data, rain_data, location):
    months = np.arange(1, 13)
    months_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    
    # Create a new figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot temperature on the left Y-axis
    ax1.plot(months, temp_data, 'r-', label=f"Mean Temperature (°C) - {location}")
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Temperature (°C)', color='red')
    ax1.tick_params(axis='y')
    ax1.set_ylim(0, 80) #adjust the limits to fit the data
    # Set x-axis labels
    ax1.set_xticks(months)  # Use month numbers as tick positions
    ax1.set_xticklabels(months_labels)  # Display single-letter month names
    
    # Create another Y-axis for rainfall
    ax2 = ax1.twinx()
    ax2.plot(months, rain_data, 'b-', label=f"Rainfall (mm) - {location}")
    ax2.set_ylabel('Rainfall (mm)', color='blue')
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, 150)
    
    # Title and formatting
    ax1.set_title(f'Climate Diagram of {location}', fontsize=16, fontweight='bold')
    
    # Show the grid and labels
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"Climate_Diagram_{location}.png")
    plt.show()

# Main execution
def main():
    
    # Read data for each location and process
    location = LOC_NAME
    data = read_text(location)
    
    # Evaluate sun data for each location
    evaluate_sun_data(data, location)
    
    # Calculate monthly averages of temperature and accumulated rainfall
    temp, rain = calculate_month(data)
    
    # Plot the climate diagram for all locations
    plot_climate_diagram(temp, rain, location)

# Call the main function to run the script
if __name__ == "__main__":
    main()

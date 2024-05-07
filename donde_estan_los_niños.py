
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 6)
excel_file = "ejercicioAI.xlsx"
excel2="Momento Favorito_Odiado.xlsx"
sheet_name1 = "datosPct2"
sheet_name2 = "renta2"
#ideal_vector=[0.23,0.02,0.02,0.03,0.10,0.16,0.15,0.115,0.02,0.0172,0.0098,0.023,0.03,0.02,0.02,0.01,0.01,0.01,0.0012,0.0038,0]
ideal_vector=[0.17,0.06,0.06,0.06,0.06,0.11,0.10,0.06,0.04,0.05,0.05,0.045,0.04,0.02,0.02,0.01,0.01,0.02,0.0012,0.0088,0.005]
minRENTA=20000
maxRENTA =27000

def donde_estan_los_niños(excel_file,sheet_name1,sheet_name2,ideal_vector,minRENTA,maxRENTA,excel2):

    # Step 2: Load Excel Data

    dtype = {
        "SeccionCensal": str,  # Assuming "SeccionCensal" is the first column
        # You can add more columns here with their respective data types
    }
    df = pd.read_excel(excel_file, sheet_name=sheet_name1, dtype=dtype, engine='openpyxl')


    # Step 3: Check the structure of the DataFrame
    #print(df.head())  # Check the first few rows to see the structure

    def find_closest_rows(input_vector, dataset):
        relevant_data = dataset.iloc[:, 2:]
        # Compute the difference between each row in the dataset and the input vector
        differences = relevant_data.sub(input_vector, axis=1)
        
        # Compute the Euclidean norm of each difference vector
        norms = np.linalg.norm(differences, axis=1)
        
        # Add the computed norms as a new column "F-of-M" to the dataset
        dataset["F-of-M"] = norms
        
        # Sort the dataset by the "F-of-M" column and get the 10 rows with the lowest values
        closest_rows = dataset.sort_values(by="F-of-M")#.head(10)
        
        return closest_rows

    
    closest_rows = find_closest_rows(ideal_vector, df)
    #print(closest_rows)


    def plot_funnel_chart(dataset):
        for i in range(0,1):
            # Find the section with the lowest F-of-M value
            #min_f_of_m_row = dataset.loc[dataset["F-of-M"].idxmin()]
            min_f_of_m_row = dataset.iloc[i]
            
            # Extract the data for the section with the lowest F-of-M value (from column 2 onwards)
            section_data = min_f_of_m_row.iloc[2:24].astype(float)
            
            # Create a horizontal bar chart (funnel chart)
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(section_data)), section_data, color='skyblue')
            plt.yticks(range(len(section_data)), section_data.index)  # Set y-axis labels to column names
            plt.xlabel("Value")
            plt.ylabel("Column")
            plt.title("Funnel Chart for Section with Lowest F-of-M Value")
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.show()
    #plot_funnel_chart(closest_rows)

    def plot_funnel_chart2(dataset):
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))  # Create a 5x2 grid of subplots
        
        for i in range(8):
            row = i // 2  # Determine the row index for subplot
            col = i % 2   # Determine the column index for subplot
            
            min_f_of_m_row = dataset.iloc[i]
            section_data = min_f_of_m_row.iloc[2:24].astype(float)
            
            ax = axes[row, col]  # Get the axis for the current subplot
            ax.barh(range(len(section_data)), section_data, color='skyblue')
            ax.set_yticks(range(len(section_data)))
            ax.set_yticklabels(section_data.index)
            ax.set_xlabel("Value")
            ax.set_ylabel("Column")
            ax.set_title(f"Funnel Chart {i+1}")
            ax.grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()




    dtype2 = {
        "Secciones": str,  # Assuming "SeccionCensal" is the first column
        # You can add more columns here with their respective data types
    }
    renta= pd.read_excel(excel_file, sheet_name=sheet_name2, dtype=dtype2,na_values=['', ' ', 'NA'], engine='openpyxl')
    renta_filled = renta.ffill()
    #print(renta_filled.head())
    renta_numeric = renta_filled[renta_filled['Secciones'].str.contains('^\d+$')]
    merged_df = closest_rows.merge(renta_numeric, left_on=closest_rows.columns[0], right_on=renta_numeric.columns[0], how='left')
    # Convert the last column to numeric format
    merged_df.iloc[:, -1] = pd.to_numeric(merged_df.iloc[:, -1], errors='coerce')

    filtered_df = merged_df[(merged_df.iloc[:, -1] >= minRENTA)&(merged_df.iloc[:, -1]<=maxRENTA )]

    dtype3 = {
        "CUSEC": str,  # Assuming "SeccionCensal" is the first column
        # You can add more columns here with their respective data types
    }
    Momento= pd.read_excel(excel2, dtype=dtype3,na_values=['', ' ', 'NA'], engine='openpyxl')
    Momento.drop(Momento.columns[1], axis=1, inplace=True)
    finalDB=filtered_df.merge(Momento,left_on=filtered_df.columns[0], right_on=Momento.columns[0], how='left')

    finalDB=finalDB[(finalDB.iloc[:,-2]!= "--") & (finalDB.iloc[:,-2]!= "-")] 
    
    # Cluster by Momento 
    grouped = finalDB.groupby(finalDB.columns[-2])

    
    for group_name, group_df in grouped:
        if group_name == "+" or group_name == "++":
            print("Cluster:", group_name)
            print(group_df.iloc[:,[0,-6,-4,-2,-1]])
            print("\n")
        
            

    plot_funnel_chart2(finalDB)
    # Añadir localizacion
    Location=pd.read_excel(excel_file, sheet_name="localizacion",dtype=str, engine='openpyxl')
    finalDB=finalDB.merge(Location,left_on=finalDB.columns[0], right_on=Location.columns[0], how='left')
    def create_bracket(row):
        return f"({row['latitud']}, {row['longitud']})"
    finalDB['Coords'] = finalDB.apply(create_bracket, axis=1)
    
    
    #finalDB["Coords"]=(float(finalDB.iloc[:,-1].replace(',', '.')),float(finalDB.iloc[:,-2].replace(',', '.')))
    #finalDB.drop(finalDB.columns[-2], axis=1, inplace=True)
    #finalDB.drop(finalDB.columns[-2], axis=1, inplace=True)

    #condition = finalDB.iloc[:, -9] > finalDB.iloc[0, -9] + 0.03
    #subset=finalDB.iloc[0:condition.idxmax(),:]
    #row=subset.loc[subset["(0,4)"].idxmax()]
    #row=row.T
    return [print(finalDB.iloc[:,[0,2,-9,-7,-5,-1]]),]

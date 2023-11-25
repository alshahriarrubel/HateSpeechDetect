import os
import pandas as pd

def binaryClassification(input_file1, input_file2, input_file3, output_file):
	print('Processing...\n')	
	# Read the CSV files
	df1 = pd.read_csv(input_file1, header=None, skiprows=1)
	df2 = pd.read_csv(input_file2, header=None, skiprows=1)
	df3 = pd.read_csv(input_file3, header=None, skiprows=1)

	# Concatenate the DataFrames
	result_df = pd.concat([df1, df2, df3])
	result_df.columns = ['class', 'tweet']
	# Write the result to a new CSV file
	result_df.to_csv(output_file, index=False)
	print('Completed!\n')


if __name__ == '__main__':
	input_file1 = os.path.join(os.getcwd(), 'Data/AHSD_binaryclass_updated.csv')
	input_file2 = os.path.join(os.getcwd(), 'Data/MHS_binaryclass.csv')
	input_file3 = os.path.join(os.getcwd(), 'Data/HATEX_binaryclass.csv')
	output_file = os.path.join(os.getcwd(), 'Data/Merged_binaryclass.csv')
	binaryClassification(input_file1, input_file2, input_file3, output_file);

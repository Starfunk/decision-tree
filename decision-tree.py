import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import operator

df = pd.read_csv("titanic_medium_b.csv")
# ~ df_1 = df
# ~ df_2 = df

#Drop row 0 from df and create new df
df_1 = df.drop(df.index[0])


#Create new df with only rows of Survived with 1
df_2 = df[df.Survived != 0]

#Returns a list of the column headings.
list_1 = df.columns.values.tolist()

#Print column 1's name
print(df.iloc[:,1].name)


#Checks to see what the key represents. Returns the first letter of the key	
#Where 0 is Pclass, 1 is sex, 2 is port.
def name_check(string):
	if string[0] == 'P':
		return 0
	elif string[0] == 'S':
		return 1
	elif string[0] == 'E':
		return 2

#Compute the entropy scores of each column and put them 
#into a dictionary.
def indiv_entropy(df,rows,cols):
	entropy_scores_P = {}
	entropy_scores_S = {}
	entropy_scores_E = {}
	survived = df.iloc[:,0]
	survived_list = list(survived)
	for i in range(1,cols):
		match_array = []
		col = df.iloc[:,i]
		col_list = list(col)
		feature = name_check(col.name)
		
		for j in range(rows):
			if col_list[j] == 1:
				match_array.append(survived_list[j])
		ones = match_array.count(1)
		zeros = rows - ones
		entropy_1 = -(ones/rows) * np.log2(ones/rows)
		entropy_2 = -(zeros/rows) * np.log2(zeros/rows)
		entropy = entropy_1 + entropy_2
		if feature == 0:
			entropy_scores_P[df.iloc[:,i].name] = entropy
		elif feature == 1:
			entropy_scores_S[df.iloc[:,i].name] = entropy
		elif feature == 2:
			entropy_scores_E[df.iloc[:,i].name] = entropy
			
	entropy_scores = [entropy_scores_P, entropy_scores_S, entropy_scores_E]
	return entropy_scores

#Compute the entropy of the current state	
def state_entropy(df,rows):
	survived = df.iloc[:,0]
	ones = 0
	for i in survived:
		 ones += i
	zeros = rows - ones
	entropy_1 = -(ones/rows) * np.log2(ones/rows)
	entropy_2 = -(zeros/rows) * np.log2(zeros/rows)
	entropy = entropy_1 + entropy_2	 
	return entropy

#Returns reduced dataframe for 'yes' case
def reduced_df_y(feature, df):
	df_2 = None
	if feature == 'Pclass_first':
		df_2 = df[df.Pclass_first != 0]
	elif feature == 'Pclass_second':
		df_2 = df[df.Pclass_second != 0]
	elif feature == 'Pclass_third':
		df_2 = df[df.Pclass_third != 0]
	elif feature == 'Sex_female':
		df_2 = df[df.Sex_female != 0]
	elif feature == 'Sex_male':
		df_2 = df[df.Sex_male != 0]
	elif feature == 'Embarked_C':
		df_2 = df[df.Embarked_C != 0]
	elif feature == 'Embarked_Q':
		df_2 = df[df.Embarked_Q != 0]
	elif feature == 'Embarked_S':
		df_2 = df[df.Embarked_S != 0]
	return df_2	
	
#Returns reduced dataframe for 'no' case	
def reduced_df_n(feature, df):
	df_2 = None
	if feature == 'Pclass_first':
		df_2 = df[df.Pclass_first != 1]
	elif feature == 'Pclass_second':
		df_2 = df[df.Pclass_second != 1]
	elif feature == 'Pclass_third':
		df_2 = df[df.Pclass_third != 1]
	elif feature == 'Sex_female':
		df_2 = df[df.Sex_female != 1]
	elif feature == 'Sex_male':
		df_2 = df[df.Sex_male != 1]
	elif feature == 'Embarked_C':
		df_2 = df[df.Embarked_C != 1]
	elif feature == 'Embarked_Q':
		df_2 = df[df.Embarked_Q != 1]
	elif feature == 'Embarked_S':
		df_2 = df[df.Embarked_S != 1]
	return df_2
	
def build_tree(df,prev_Q):
	
	print("The previous node was: " + prev_Q)
	#Recording # of rows of the data set
	rows = df.shape[0]

	#Recording # of rows of the data set
	cols = df.shape[1]
		
	if cols == 3:
		col_1_name = df.iloc[:,1].name

		
		df_reduce_1 = reduced_df_y(col_1_name,df)
		
		col_1 = list(df_reduce_1.iloc[:,0])
		
		ones_1 = col_1.count(1)
		zeros_1 = col_1.count(0)
		
		if ones_1 > zeros_1:
			print("For: " + col_1_name + " - survived!")
			print("ones: " + str(ones_1) + " zeros: " + str(zeros_1))
		else:
			print("For: " + col_1_name + " - died.")
			print("ones: " + str(ones_1) + " zeros: " + str(zeros_1))
		
		col_2_name = df.iloc[:,2].name
		df_reduce_2 = reduced_df_y(col_2_name,df)
		col_2 = list(df_reduce_2.iloc[:,0])
		ones_2 = col_2.count(1)
		zeros_2 = col_2.count(0)
	
		if ones_2 > zeros_2:
			print("For: " + col_2_name + " - survived!")
			print("ones: " + str(ones_2) + " zeros: " + str(zeros_2))
		else:
			print("For: " + col_2_name + " - died.")
			print("ones: " + str(ones_2) + " zeros: " + str(zeros_2))
		
		print()
		return

	#Create a dictionary of the info gain scores
	entropy_scores = indiv_entropy(df,rows,cols)
	entropy = state_entropy(df,rows)
	class_sum = sum(entropy_scores[0].values())
	sex_sum = sum(entropy_scores[1].values())
	embark_sum = sum(entropy_scores[2].values())
	
	counter1 = len(entropy_scores[0])
	counter2 = len(entropy_scores[1])
	counter3 = len(entropy_scores[2])

	info_gain1 = None
	info_gain2 = None
	info_gain3 = None
	
	if counter1 != 0:
		info_gain1 = entropy - class_sum/counter1
	if counter2 != 0:
		info_gain2 = entropy - sex_sum/counter2
	if counter3 != 0:
		info_gain3 = entropy - embark_sum/counter3
		
	info_gain = {}
	if info_gain1 != None:
		info_gain['Passenger'] = info_gain1
	if info_gain2 != None:
		info_gain['Sex'] = info_gain2
	if info_gain3 != None:
		info_gain['Embark'] = info_gain3
	
	
	max_info = max(info_gain.items(), key=operator.itemgetter(1))
	
	feature = name_check(max_info[0])
	
	column = max_info[0]

	category_value_max = info_gain[max_info[0]]

		
	list_items = list(entropy_scores[feature].items())
	sorted_list = sorted(list_items, key=lambda val: val[1])
	feature_max = sorted_list[0][0]
	
	#For yes branch
	reduce_df_1 = reduced_df_y(feature_max,df)
	#For no branch	
	reduce_df_2 = reduced_df_n(feature_max,df)
	
	
	q_y = "If yes: " + feature_max + "."
	q_n = "If no: " + feature_max + "."
	
	print("This nodes feature is: " + feature_max)
	print(feature_max + " entropy: " + str(sorted_list[0][1]))
	print("The highest information gain option: " + max_info[0] + \
		" = " + str(category_value_max))
	print()
	
	if counter1 > 2:
		reduce_df_1 = reduce_df_1.drop(columns=[sorted_list[0][0]])
		reduce_df_1 = reduce_df_1.drop(columns=[sorted_list[1][0]])
		reduce_df_1 = reduce_df_1.drop(columns=[sorted_list[2][0]])
			
		reduce_df_2 = reduce_df_2.drop(columns=[sorted_list[0][0]])
		
		#Yes branch
		build_tree(reduce_df_1,q_y)
		#No branch
		build_tree(reduce_df_2,q_n)
			
	elif counter1 == 2:
		reduce_df_1 = reduce_df_1.drop(columns=[sorted_list[0][0]])
		reduce_df_1 = reduce_df_1.drop(columns=[sorted_list[1][0]])
			
		reduce_df_2 = reduce_df_2.drop(columns=[sorted_list[0][0]])
		reduce_df_2 = reduce_df_2.drop(columns=[sorted_list[1][0]])
		
		#Yes branch
		build_tree(reduce_df_1,q_y)
		#No branch
		build_tree(reduce_df_2,q_n)
		
	elif counter1 == 1:
		print("THIS SHOULD NOT BE EXECUTING!")
		
q = 'top node'	
print(build_tree(df,q))	



import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import time
from nptdms import TdmsFile
from npfeintool import SegHub
#from .segmentation import SegHub
from .cleaner import CLI
import json
import  glob
import fnmatch
import shutil





class CON:
    @staticmethod
    def setup_directory(experiment, revision):
        '''
        Parses the TDMS file name located in the directory to an sqlite table named like the tdms file.
        '''
        exp_id = experiment + "_" + revision
        rev = revision
        
        os.makedirs("./data/" + exp_id + "/plain/" + rev )
        os.makedirs("./data/" + exp_id + "/segmented/" + rev)
        os.makedirs("./data/" + exp_id + "/combined/" + rev)
        os.makedirs("./data/" + exp_id + "/cleaned/"  + rev)
        os.makedirs("./data/" + exp_id + "/subsegmented/" + rev)
        os.makedirs("./data/" + exp_id + "/tdms/" + rev)
        
    @staticmethod
    def get_names(directory):
        '''
        Extract the names of the .tdms or .json files in a directory.
        '''
        

        file_names = []
        print(directory)
        for root, dirs, files in os.walk(directory):
            print(files)
            for filename in files:
                if ("tdms" in filename):
                    name_temp = filename.replace(".tdms", "").replace("-", "_").replace(".db", "")
                    os.rename(directory+filename,directory+name_temp+".tdms")
                    file_names.append(name_temp)

                elif ("db" in filename): 
                    name_temp = filename.replace(".tdms", "").replace("-", "_").replace(".db", "")
                    os.rename(directory+filename,directory+name_temp+".db")
                    file_names.append(name_temp)

                elif ("json" in filename): 
                    file_names.append(filename)

        return file_names
    
    
    @staticmethod
    def tdms_to_json(directory, target_directory, name):

        '''
        Parses the TDMS file name located in the directory to json file named like the tdms file
        '''

        print("Parsing: " + directory + name + ".tdms")

        # load in data, take the TDMS data type as example
        tdms_file = TdmsFile(directory + name + ".tdms")
        groups = tdms_file.groups()

        df_list = []
        #df_test = pd.DataFrame()
        df_list.append(pd.DataFrame())
        for group in groups:

            print(group)

            for channel in tdms_file.group_channels(group):

            	
            	try:
            		chan_name = str(channel).split("'/'")[1].replace("'>", "")
            		data = tdms_file.channel_data(group, chan_name)
            		chan_name = chan_name.replace("(", " ").replace(")", " ").replace("  ", " ").replace(" ", "_")
            		
            		df_test = pd.DataFrame()

            		#print(chan_name)
            		# print(data)
            		#time.sleep(0.5)
            		df_test[chan_name] = chan_name #data
            		df_list[-1][chan_name] = chan_name #data
            		#df_test.to_json(target_directory + chan_name + ".json", orient='split')
            		# print(chan_name)
            		# print(data)
            		#df_test.to_json(target_directory + chan_name + ".json", orient='split')
            		#df_test.to_json(target_directory  + chan_name +  '_' + name   + ".json", orient='split')
            		with open(target_directory  + chan_name +  '_' + name   + ".json", 'w') as fp:

            			json.dump({chan_name:data.tolist()}, fp)
                        
                        
            	except:
            		print("An Error Occured at: X  !")
            		continue
            
        #df_list[-1].to_json(target_directory + name + ".json")
        #df_test.to_json(target_directory + name + ".json", orient='split')
       # df_all = pd.concat(df_list, axis=1)
        #df_all.to_json(target_directory + name + ".json")
        #df_test.to_json(target_directory + name + ".json")
        
        
        
    @staticmethod
    def get_punch_index(directory, revision, file_names, sensor="Stempel_1"):
        '''
        gets the number of punches from each data, it tells when the machine stopped and restarted 
        '''
        print(directory + "plain/" + revision + "/")
        names = CON.get_names(directory + "plain/" + revision + "/")
        
        #conn_comb = sqlite3.connect(directory + "metadata/" + data_base + "_total_punches.db")
        total_punch_list = []
        print(names)
        #fetch all dataframes
        for name in names:
            print(name)
            #conn_1 = sqlite3.connect(directory + "plain/" + revision + "/" + name + ".db")
            df_temp = pd.read_json(directory + "plain/" + revision + "/" + name, orient='split')
            #df_names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn_1)
            #df_temp = pd.read_sql_query("select " + sensor + " from " + df_names['name'][0], conn_1)
            peaks, _ = find_peaks(df_temp[sensor], height=70000, distance=8000)
            total_punch_list.append(len(peaks))
        
        df_punch_index = pd.DataFrame()
        for total in total_punch_list:
           df_punch_index = pd.concat([df_punch_index, pd.Series(range(total))], ignore_index=True)
        
        df_punch_index = df_punch_index.rename(columns={0: "real_index"})
        #write to data base file, table is named as the database
        #df_punch_index.to_sql(data_base, conn_comb, if_exists="replace")
        df_punch_index.to_json(directory + "restart_metadata_" + revision + ".json")
        
        
        
        
                
    @staticmethod                
    def combine_sensor(directory, experiment, revision):
        
        '''
        Moves each sensor data to their respective directories and merges sensor data in the directories. 
        Does same job of the previous py script CON.combine_rework and CON.segmented_database_rework.
        '''
        
        exp_id = experiment + "_" + revision
        #rev = revision

        
        #directories 
        
        directory_work = directory + 'data/' + exp_id + '/' + 'plain' +'/' + revision +'/'
        export_dir = directory + 'data/' + exp_id + '/' + 'combined' +'/' + revision +'/'
        segmented_dir = directory + 'data/' + exp_id + '/' + 'segmented' +'/' + revision +'/'
        file_names = CON.get_names(directory_work)
        
        sensor_dir = ["Stempel_1", "Stempel_2", "Stempel_3", "Stempel_4", 
                      "Gegenhalter", "Niederhalter_1", "Niederhalter_2", "Niederhalter_4", 
                      "Bolzen_LU", "Bolzen_RU", "Bolzen_LO", "Bolzen_RO", 
                      "Position_NH", "Position_Ma", "AE_oben", "AE_unten", 
                      "Sound","Beschleunigung_oben", "Beschleunigung_unten", "DigitalIn", "Untitled"]
        
        for sensor in sensor_dir:
            os.makedirs(export_dir + sensor + "/")
        
        
        # moving the files to their respective dir
        for file in file_names:
            if 'Stempel_1' in file:
                 shutil.move(directory_work + file, export_dir + 'Stempel_1'+ '/' )
                
            elif 'Stempel_2' in file:
                shutil.move(directory_work + file, export_dir + 'Stempel_2' + '/' )
    
            elif 'Stempel_3' in file:
                shutil.move(directory_work + file, export_dir + 'Stempel_3'+ '/' )
    
            elif 'Stempel_4' in file:
                shutil.move(directory_work + file, export_dir + 'Stempel_4'+ '/' )
    
            elif 'Gegenhalter' in file:
                shutil.move(directory_work + file, export_dir + 'Gegenhalter'+ '/' )
    
            elif 'Niederhalter_1' in file:
                shutil.move(directory_work + file, export_dir + 'Niederhalter_1'+ '/' )
    
            elif 'Niederhalter_2' in file:
                shutil.move(directory_work + file, export_dir + 'Niederhalter_2'+ '/' )
    
            elif 'Niederhalter_4' in file:
                shutil.move(directory_work + file, export_dir + 'Niederhalter_4'+ '/' )
                
            elif 'Bolzen_LU' in file:
                shutil.move(directory_work + file, export_dir + 'Bolzen_LU'+ '/' )
    
            elif 'Bolzen_RU' in file:
                shutil.move(directory_work + file, export_dir + 'Bolzen_RU'+ '/' )
    
            elif 'Bolzen_RO' in file:
                shutil.move(directory_work + file, export_dir + 'Bolzen_RO'+ '/' )
    
            elif 'Bolzen_LO' in file:
                shutil.move(directory_work + file, export_dir + 'Bolzen_LO'+ '/' )
    
            elif 'Position_NH' in file:
                shutil.move(directory_work + file, export_dir + 'Position_NH'+ '/' )
    
            elif 'Position_Ma' in file:
                shutil.move(directory_work + file, export_dir + 'Position_Ma'+ '/' )
    
            elif 'AE_oben' in file:
                shutil.move(directory_work + file, export_dir + 'AE_oben'+ '/' )
                
            elif 'AE_unten' in file:
                shutil.move(directory_work + file, export_dir + 'AE_unten'+ '/' )
            
            elif 'Sound' in file:
                shutil.move(directory_work + file, export_dir + 'Sound'+ '/' )
            
            elif 'Beschleunigung_oben' in file:
                shutil.move(directory_work + file, export_dir + 'Beschleunigung_oben'+ '/' )
            
            elif 'Beschleunigung_unten' in file:
                shutil.move(directory_work + file, export_dir + 'Beschleunigung_unten'+ '/' )
            
            elif 'DigitalIn' in file:
                shutil.move(directory_work + file, export_dir + 'DigitalIn'+ '/' )
            
            elif 'Untitled' in file:
                shutil.move(directory_work + file, export_dir + 'Untitled'+ '/' )
                
                
                
        #converting the files to df and then merging them 
        for sensor in sensor_dir:
            src_files = os.listdir(export_dir + sensor + '/')
            
            #src_files = os.listdir(export_dir +'Stempel_2' + '/')
            result= []
            for file in src_files:
                df_temp = pd.read_json(export_dir + sensor+ '/'+ file)
                result.append(df_temp)
                merge_result = pd.concat(result, axis=1)
                merge_result.to_json(segmented_dir + sensor + '.json', orient='split')
                
    @staticmethod
    def converting(experiment, revision, directory):
        
        '''
        
        '''
        
        '''  
        ----------------------------- INIT-----------------------------------
        '''
        # experiment = "SS"
        # revision = "FW2"
        
        exp_id = experiment + "_" + revision
        directory_work = directory + 'data/' + exp_id + '/'
        
        
        '''  
        ----------------------------- INIT-----------------------------------
        '''

        #CON.setup_directory(experiment, revision)


        '''  
        ------------------------- CONVERTING ---------------------------------
        '''
        
        file_names = CON.get_names(directory + 'data/' + exp_id + '/' + 'tdms' +'/' + revision +'/')
        print(file_names)
        
        print("./data/" + experiment + "/tdms/" + revision + "/")
        print("Begin converting .tdms files to .json files.")
        
        for name in file_names: 
            CON.tdms_to_json (directory = directory_work + 'tdms' +'/' + revision +'/',
                              target_directory = directory_work + "plain/" + revision + "/",
                              name = name)
        
        print("Finished converting .tdms files to .json files!")
        
        
        '''
        ------------------------- METADAT ---------------------------------
        '''
        #CON.get_punch_index(working_dir, revision, "name", "name")

        
        '''  
        ------------------------- COMBINING ˆ& SEGMENTING ---------------------------------
        
        combining related sensors of all files together in single sensor json files
        '''
        
        print("Begin combining and segmenting...")
        
        CON.combine_sensor(directory = directory, experiment=experiment, revision=revision)
        
        print("Finished combining and segmenting!")
        print ("Finished running .converting on " + exp_id + " files.")
        
        
                
                
                
    
                
            
             
    
        
        
        
        
        
        
    
    





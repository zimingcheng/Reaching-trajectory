# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 19:49:45 2020

@author: czm19
"""

import sqlite3
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance
from data_total_plot_0220 import interpolate_plot_zi
from sklearn.model_selection import train_test_split
from tensorflow import keras

def run_query(db, q, args=None):
    
    """
    Return the results of running query q with arguments args on
    database db.
    """

    run_con = sqlite3.connect(db)
    run_cur = run_con.cursor()

    if args is None:  # if no arguments are given:
        run_cur.execute(q)  # implement

    else:  # if arguments are give:
        run_cur.execute(q, args)

    run_data = run_cur.fetchall()
    run_cur.close()
    run_con.close()
    return run_data

def get_raw_data(filename):
    """
    Return the data from the csv file to a list of list of objects
    """
    instream = open(filename)
    reader = csv.reader(instream)
    header_data = [line for line in reader]
    raw_data = header_data[13:]
    return raw_data

def get_par_id(filename):
    """
    Return participant ID from the csv file
    """
    instream = open(filename)
    reader = csv.reader(instream)
    header_data = [line for line in reader]
    p_num = int(header_data[0][1])
    return p_num

def get_data(raw_data):
    """
    Return the cleaned raw_data by removing the track, changing str to int or
    float if applicable, and removing empty strings
    """
    data = []
    for line in raw_data:
        temp = line[:12]
        i = 12
        trace = []
        timestamp = []
        while line[i] != "" and line[i][1:] != "":
            point = line[i]
            if point[0] == ' ':
                point = point[1:]
            x = float(point.split(' : ')[0])
            y = float(point.split(' : ')[1].split(' | ')[0])
            time = int(point.split(' : ')[1].split(' | ')[1])
            trace.append((x,y))
            timestamp.append(time)
            i += 1
        temp.append(trace)
        temp.append(timestamp)
        data.append(temp)
    return data

def create_result_table(my_db):
    """
    Create the result table
    """
    con = sqlite3.connect(my_db)
    cur = con.cursor()

    cur.execute('''CREATE TABLE Result (ParID INTEGER, TrialID INTEGER, Left TEXT
    , Right TEXT, Hit TEXT, Correct INTEGER, StartTime INTEGER, EndTime INTEGER
    , Feedback TEXT, Disparity REAL, Length REAL, Order_ INTEGER)''')

    cur.close()
    con.commit()
    con.close()

def populate_result_table(my_db, filename):
    """
    Populate the result table if the current participant ID is not already
    in the database
    """
    par_id = get_par_id(filename)
    raw_data = get_raw_data(filename)
    data = get_data(raw_data)

    con = sqlite3.connect(my_db)
    cur = con.cursor()

    p_num_exist = run_query(my_db, 'SELECT COUNT(ParID) FROM Result WHERE ParID = ?',
                            (par_id,))

    if p_num_exist[0][0] == 0:
        for line in data:
            if line[0] != '':
                trial_id = int(line[0])
                left = line[1]
                right = line[2]
                hit = line[5]
                correct = line[6]
                endtime = line[7]
                if endtime != "undefined":
                    endtime = int(line[7])
                starttime = line[13][0]
                feedback = line[8]
                disparity = line[9]
                if disparity != "undefined":
                    disparity = float(line[9])
                length = line[10]
                if length != "undefined":
                    length = float(line[10])
                order = line[11]
                cur.execute('''INSERT INTO Result VALUES(?, ?, ?, ?, ?, ?, ?, 
                            ?, ?, ?, ?, ?)''',
                    (par_id, trial_id, left, right, hit, correct, starttime, 
                     endtime, feedback, disparity, length, order))
    print(f"===============result data added for {filename}===============")
    cur.close()
    con.commit()
    con.close()

def create_input_table(my_db):
    """
    Create input table
    """
    con = sqlite3.connect(my_db)
    cur = con.cursor()

    cur.execute('''CREATE TABLE Input (TrialID INTEGER, SetID INTEGER, 
                LeftID INTEGER, LeftName TEXT, RightID INTEGER, RightName TEXT,
                Visual REAL, Conceptual REAL, Condition TEXT)''')
    
    cur.close()
    con.commit()
    con.close()
       
def populate_input_table(my_db, filename):
    """
    Populate the input table
    """
    instream = open(filename)
    reader = csv.reader(instream)
    header_data = [line for line in reader]
    data = header_data[1:]
    
    con = sqlite3.connect(my_db)
    cur = con.cursor()
    
    for line in data:
        trial_id = int(line[0])
        set_ = int(line[1])
        left_id = int(line[2])
        left_name = line[3]
        right_id = int(line[5])
        right_name = line[6]
        vis_sim = float(line[8])
        con_sim = float(line[9])
        condition = line[10]
        cur.execute('''INSERT INTO Input VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (trial_id, set_, left_id, left_name, right_id, right_name,
             vis_sim, con_sim, condition))
    
    print(f"===============input data added from {filename}===============")
    cur.close()
    con.commit()
    con.close()    
       
def del_input_table(my_db):
    """
    Clear the input table
    """
    con = sqlite3.connect(my_db)
    cur = con.cursor()
    cur.execute("DELETE FROM Input")
    cur.close()
    con.commit()
    con.close() 
    
def par_info(my_db, par_list, all_output):
    for par_id in par_list: 
        if par_id not in all_output:
            output = {}
            output['ptp'] = run_query(my_db, '''SELECT ParID FROM Result JOIN
                                          Input ON Result.TrialID = Input.TrialID WHERE
                                          ParID = ?''', (par_id,))
            output['rating'] = [ptp_list[par_id]] * len(output['ptp'])
            output['trialID'] = run_query(my_db, '''SELECT Result.TrialID FROM Result JOIN
                                          Input ON Result.TrialID = Input.TrialID WHERE
                                          ParID = ?''', (par_id,))
            output['trial_order'] = run_query(my_db, '''SELECT Order_ FROM Result JOIN
                                          Input ON Result.TrialID = Input.TrialID WHERE
                                          ParID = ?''', (par_id,))
            output['setID'] = run_query(my_db, '''SELECT SetID FROM Result JOIN
                                          Input ON Result.TrialID = Input.TrialID WHERE
                                          ParID = ?''', (par_id,))
            output['left'] = run_query(my_db, '''SELECT Left FROM Result JOIN
                                          Input ON Result.TrialID = Input.TrialID WHERE
                                          ParID = ?''', (par_id,))
            output['right'] = run_query(my_db, '''SELECT Right FROM Result JOIN
                                          Input ON Result.TrialID = Input.TrialID WHERE
                                          ParID = ?''', (par_id,))
            output['correct'] = run_query(my_db, '''SELECT Correct FROM Result 
                                            JOIN Input ON Result.TrialID = 
                                            Input.TrialID WHERE ParID = ?''', (par_id,))
            output['visual_similarity'] = run_query(my_db, '''SELECT Visual FROM Result 
                                            JOIN Input ON Result.TrialID = 
                                            Input.TrialID WHERE ParID = ?''', (par_id,))
            output['conceptual_similarity'] = run_query(my_db, '''SELECT Conceptual FROM Result 
                                            JOIN Input ON Result.TrialID = 
                                            Input.TrialID WHERE ParID = ?''', (par_id,))
            output['condition'] = run_query(my_db, '''SELECT Condition FROM Result 
                                            JOIN Input ON Result.TrialID = 
                                            Input.TrialID WHERE ParID = ?''', (par_id,))
            output['disparity'] = run_query(my_db, '''SELECT Disparity FROM Result 
                                            JOIN Input ON Result.TrialID = 
                                            Input.TrialID WHERE ParID = ?''', (par_id,))
            output['length'] = run_query(my_db, '''SELECT Length FROM Result 
                                            JOIN Input ON Result.TrialID = 
                                            Input.TrialID WHERE ParID = ?''', (par_id,))
            output['start_time'] = run_query(my_db, '''SELECT StartTime FROM Result 
                                            JOIN Input ON Result.TrialID = 
                                            Input.TrialID WHERE ParID = ?''', (par_id,))
            output['end_time'] = run_query(my_db, '''SELECT EndTime FROM Result 
                                            JOIN Input ON Result.TrialID = 
                                            Input.TrialID WHERE ParID = ?''', (par_id,))
            for key in output:
                lst = output[key]
                if key == 'stimuli':
                    lst_new = [set(item) for item in lst]
                else:
                    lst_new = [item[0] for item in lst]
                output[key] = lst_new
            all_output[par_id] = output
        else:
            print('ptp already in info')
    return all_output

def initiate(my_db, input_):
    create_result_table(my_db)
    create_input_table(my_db)
    populate_input_table(my_db, input_)

def populate(my_db, ptp_list):
    for i in ptp_list:
        result = f'{i}.csv'
        populate_result_table(my_db, result)
        
def graph(df):
    plt.scatter(df['conceptual'], df['disparity'])
    con_disparity = np.polyfit(df['conceptual'], df['disparity'], 1)
    k = con_disparity[0]
    b = con_disparity[1]
    plt.plot(df['conceptual'], k * df['conceptual'] + b)
    plt.xlabel('conceptual similarity')
    plt.ylabel('disparity')
    plt.show()
    plt.scatter(df['visual'], df['disparity'])
    vis_disparity = np.polyfit(df['visual'], df['disparity'], 1)
    k = vis_disparity[0]
    b = vis_disparity[1]
    plt.plot(df['visual'], k * df['visual'] + b)
    plt.xlabel('visual similarity')
    plt.ylabel('disparity')
    plt.show()
      
def speed_spec(point_list, time_list):
    # min_, imin = 0, 0
    dist = []
    for i in range(1, len(point_list)):
        dist.append(distance.euclidean(point_list[i-1], point_list[i]))
    max_, min_ = max(dist), min(dist)
    imax, imin = dist.index(max_), dist.index(min_)
    tmax, tmin = time_list[imax], time_list[imin]
    up = distance.euclidean(point_list[0], point_list[imax])
    return max_, tmax, min_, tmin, up

def get_ptp_reaches(ptp):
    data = get_data(get_raw_data(f'{ptp}.csv'))
    points = [line[12] for line in data]
    return points

def get_point_number():
    all_number = []
    for ptp in ptp_list:
        reaches = get_ptp_reaches(ptp)
        numbers = [len(reach) for reach in reaches]
        all_number.extend(numbers)
    plt.hist(all_number, bins=20)
    plt.show()
    return all_number

def get_starting_location():
    starting_locations = []
    for ptp in ptp_list:
        reaches = get_ptp_reaches(ptp)
        starts = [reach[0] for reach in reaches]
        starting_locations.extend(starts)
    x = [point[0] for point in starting_locations]
    y = [point[1] for point in starting_locations]
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    plt.gca().invert_yaxis()
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='Reds')
    plt.show()
    plt.gca().invert_yaxis()
    plt.scatter(x, y)
    plt.show()

def interpolate_reaches(n, reaches):
    new_reaches = []
    for i in range(len(reaches)):
        # print(i)
        x = [point[0] for point in reaches[i]]
        y = [point[1] for point in reaches[i]]
        interpolated = interpolate_plot_zi(n,x,y)
        new_reach = list(map(tuple, interpolated))
        new_reaches.append(new_reach)
    return new_reaches

def interpolate_all_reaches(n, all_output):
    reach_info = {}
    for ptp in all_output.keys():
        # print(ptp)
        reach_info[ptp] = {}
        reaches = get_ptp_reaches(ptp)
        new_reaches = interpolate_reaches(n, reaches)
        reach_info[ptp]['ptp'] = ptp
        reach_info[ptp]['reach_ori'] = reaches
        reach_info[ptp]['reach_int'] = new_reaches
        reach_info[ptp]['similarity'] = all_output[ptp]['condition']
        reach_info[ptp]['rating'] = all_output[ptp]['rating']
        reach_info[ptp]['conceptual'] = all_output[ptp]['conceptual']
        reach_info[ptp]['visual'] = all_output[ptp]['visual']
        reach_info[ptp]['conceptual_label'] = np.where(np.float32(reach_info[ptp]['conceptual'])<0.5,'low','high')
        reach_info[ptp]['visual_label'] = np.where(np.float32(reach_info[ptp]['visual'])<0.5,'low','high')
        reach_info[ptp]['any_sim'] = all_output[ptp]['conceptual'] + all_output[ptp]['visual']
    return reach_info

def reach_matrix(all_reaches, df):
    reach = []
    condition = []
    similarity = []
    from itertools import chain
    for ptp in all_reaches:
        condition.extend([all_reaches[ptp]['rating'][i] + '_' + 
                          all_reaches[ptp]['similarity'][i] for i in 
                          range(len(all_reaches[ptp]['similarity']))])
        similarity.extend([all_reaches[ptp]['conceptual_label'][i] + '_' +
                            all_reaches[ptp]['visual_label'][i]
                            for i in range(len(all_reaches[ptp]['conceptual_label']))])
        for line in all_reaches[ptp]['reach_int']:
            flat_line = list(chain(*line))
            reach.append(flat_line)
    return np.asarray(reach), np.asarray(condition), \
        np.asarray(similarity), np.asarray(df['reach_type'])

def PCA(reach_matrix):
    import seaborn as sns
    C = reach_matrix[0]
    fun = lambda x: int(x[0] == 'v')
    label = np.asarray(list(map(fun, reach_matrix[1])))
    num_to_text = {0:'conceptual rating', 1:'visual rating'}
    # fun = lambda x: int(len(x) > 7)
    # label = np.asarray(list(map(fun, reach_matrix[2])))
    # num_to_text = {0:'low similarity', 1:'high similarity'}
    # label = np.vectorize(num_to_text.get)(label)
    # label = reach_matrix[3]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(C)
    reduced = pca.transform(C)
    plt.figure(figsize=(10, 10))
    x = reduced[:, 0]
    y = reduced[:, 1]
    sns.scatterplot(x=x, y=y, hue=label)
    plt.show()
    
def type_preprocessing(reach_matrix, style):
    reaches = np.asarray([[line] for line in reach_matrix[0]])
    if style == 'rate_target':
        mapping = {'c_concept':0, 'c_visual':1, 'v_concept':2, 'v_visual':3}
        labels = np.asarray([mapping[label] for label in reach_matrix[1]])
        output_layer = 4
    elif style == 'rate':
        fun = lambda x: int(x[0] == 'v')
        labels = np.asarray(list(map(fun, reach_matrix[1])))
        output_layer = 2
    elif style == 'target':
        fun = lambda x: int(x[-1] == 'l')
        labels = np.asarray(list(map(fun, reach_matrix[1])))
        output_layer = 2
    elif style == 'congruent':
        fun = lambda x: int(x[0] == x[2])
        labels = np.asarray(list(map(fun, reach_matrix[1])))
        output_layer = 2
    elif style == 'reach_type':
        fun = lambda x: int(x == 'controlled')
        labels = np.asarray(list(map(fun, reach_matrix[3])))
        output_layer = 2
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(1, 200)), 
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(output_layer, activation="softmax")
        ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model, reaches, labels

def sim_pre_preprocessing(reaching_matrix):
    import random
    counts = np.unique(reaching_matrix[2], return_counts=True)
    maximum = np.min(counts[1])
    random.seed(1)
    random.shuffle(reaching_matrix[0])
    random.seed(1)
    random.shuffle(reaching_matrix[2])
    random.seed(1)
    random.shuffle(reaching_matrix[3])
    ll, lh, hl, hh = 0, 0, 0 ,0
    reduced_reaching_matrix = ([], [], [])
    further_reduced_reaching_matrix = ([], [], [])
    for i in range(len(reaching_matrix[0])):
        # if reaching_matrix[1][i][0] == rating and reaching_matrix[3][i] == reach_type:
        if reaching_matrix[2][i] == 'low_low':
            ll += 1
            if ll <= maximum:
                reduced_reaching_matrix[0].append(reaching_matrix[0][i])
                reduced_reaching_matrix[1].append(reaching_matrix[2][i])
                reduced_reaching_matrix[2].append(reaching_matrix[3][i])
                further_reduced_reaching_matrix[0].append(reaching_matrix[0][i])
                further_reduced_reaching_matrix[1].append(reaching_matrix[2][i])
                further_reduced_reaching_matrix[2].append(reaching_matrix[3][i])
        elif reaching_matrix[2][i] == 'low_high':
            lh += 1
            if lh <= maximum:
                reduced_reaching_matrix[0].append(reaching_matrix[0][i])
                reduced_reaching_matrix[1].append(reaching_matrix[2][i])
                reduced_reaching_matrix[2].append(reaching_matrix[3][i])
            if lh + hl + hh <= maximum:
                further_reduced_reaching_matrix[0].append(reaching_matrix[0][i])
                further_reduced_reaching_matrix[1].append(reaching_matrix[2][i])
                further_reduced_reaching_matrix[2].append(reaching_matrix[3][i])
        elif reaching_matrix[2][i] == 'high_low':
            hl += 1
            if hl <= maximum:
                reduced_reaching_matrix[0].append(reaching_matrix[0][i])
                reduced_reaching_matrix[1].append(reaching_matrix[2][i])
                reduced_reaching_matrix[2].append(reaching_matrix[3][i])
            if lh + hl + hh <= maximum:
                further_reduced_reaching_matrix[0].append(reaching_matrix[0][i])
                further_reduced_reaching_matrix[1].append(reaching_matrix[2][i])
                further_reduced_reaching_matrix[2].append(reaching_matrix[3][i])
        elif reaching_matrix[2][i] == 'high_high':
            hh += 1
            if hh <= maximum:
                reduced_reaching_matrix[0].append(reaching_matrix[0][i])
                reduced_reaching_matrix[1].append(reaching_matrix[2][i])
                reduced_reaching_matrix[2].append(reaching_matrix[3][i])
            if lh + hl + hh <= maximum:
                further_reduced_reaching_matrix[0].append(reaching_matrix[0][i])
                further_reduced_reaching_matrix[1].append(reaching_matrix[2][i])
                further_reduced_reaching_matrix[2].append(reaching_matrix[3][i])
    return reduced_reaching_matrix, further_reduced_reaching_matrix

def con_or_vis_preprocessing(reach_matrix):
    reach_matrix = sim_pre_preprocessing(reach_matrix)[1]
    reaches = np.asarray([[line] for line in reach_matrix[0]])
    fun = lambda x: int(len(x) > 7)
    labels = np.asarray(list(map(fun, reach_matrix[1])))
    # reach_matrix[1] is used because sim_pre_preprocessing drops the orginal 
    # reaching_matrix[1], yes, we are using the right labels
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(1, 200)), 
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(2, activation="softmax")
        ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model, reaches, labels
    
def sim_preprocessing(reach_matrix, style):
    if style == 'con_or_vis':
        model, reaches, labels = con_or_vis_preprocessing(reach_matrix)
    else:
        reach_matrix = sim_pre_preprocessing(reach_matrix)[0]
        reaches = np.asarray([[line] for line in reach_matrix[0]])
        if style == 'con_and_vis':
            mapping = {'low_low':0, 'low_high':1, 'high_low':2, 'high_high':3}
            labels = np.asarray([mapping[label] for label in reach_matrix[1]])
            output_layer = 4
        if style == 'con':
            fun = lambda x: int(x[0] == 'h')
            labels = np.asarray(list(map(fun, reach_matrix[1])))
            output_layer = 2
        if style == 'vis':
            fun = lambda x: int(x[-1] == 'h')
            labels = np.asarray(list(map(fun, reach_matrix[1])))
            output_layer = 2
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(1, 200)), 
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(output_layer, activation="softmax")
            ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    return model, reaches, labels
    
def decoder(model, reaches, labels):
    output = []
    for i in range(100):
        reach_train, reach_test, label_train, label_test = \
        train_test_split(reaches, labels, test_size=0.2) 
        model.fit(reach_train, label_train, epochs=5)
        test_loss, test_acc = model.evaluate(reach_test, label_test)
        output.append(test_acc)
    return np.array(output)

def compare_decoder(reach_matrix, kind):
    output = {}
    if kind == 'type':
        styles = ['reach_type']
        for style in styles:
            model, reaches, labels = type_preprocessing(reach_matrix, style)
            style_output = decoder(model, reaches, labels)
            output[style] = style_output
    elif kind == 'sim':
        styles = ['con_and_vis', 'con', 'vis', 'con_or_vis']
        for style in styles:
            model, reaches, labels = sim_preprocessing(reach_matrix, style)
            style_output = decoder(model, reaches, labels)
            output[style] = style_output
    elif kind == 'rate_reach_type':
        for rating in ['c', 'v']:
            for reach_type in ['controlled', 'automatic']:
                model, reaches, labels = con_or_vis_preprocessing\
                    (reach_matrix, rating, reach_type)
                style_output = decoder(model, reaches, labels)
                output[f'{rating}_{reach_type}'] = style_output
    return output

def graph_reach(reach):
    xs = [point[0] for point in reach]
    ys = [point[1] for point in reach]
    plt.gca().invert_yaxis()
    plt.scatter(xs, ys)
    plt.show()
    
def flat_to_point(reach):
    x, y = [], []
    for i in range(len(reach)):
        if i % 2 == 0:
            x.append(reach[i])
        else:
            y.append(reach[i])
    return x, y

def graph_from_matrix(reaches, c='b'):
    x = reaches[:, ::2]
    y = reaches[:, 1::2] * -1.0
    left_lst, right_lst = ([],[]), ([],[])
    left, right = [[],[]], [[],[]]
    for i in range(len(x)):
        x_temp, y_temp = x[i,:], y[i,:]
        if x_temp[-1] < 0:
            left_lst[0].append(x_temp)
            left_lst[1].append(y_temp)
        else:
            right_lst[0].append(x_temp)
            right_lst[1].append(y_temp)
    left[0], right[0], left[1], right[1] = \
        np.asarray(left_lst[0]), np.asarray(right_lst[0]),\
        np.asarray(left_lst[1]), np.asarray(right_lst[1])
    l_x_mean = np.mean(left[0], axis=0)
    l_y_mean = np.mean(left[1], axis=0)
    r_x_mean = np.mean(right[0], axis=0)
    r_y_mean = np.mean(right[1], axis=0)
    l_x_sme = np.std(left[0], axis=0)/np.sqrt(18)
    r_x_sme = np.std(right[0], axis=0)/np.sqrt(18)
    # y_sme = np.std(y, axis=0)
    l_x_low, l_x_high = l_x_mean - l_x_sme , l_x_mean + l_x_sme
    r_x_low, r_x_high = r_x_mean - r_x_sme , r_x_mean + r_x_sme
    # y_low, y_high = y_mean - y_sme, y_mean + y_sme
    plt.plot(l_x_mean, l_y_mean, linewidth=2, color=c)
    plt.fill_betweenx(l_y_mean, l_x_high, l_x_low, color=c, alpha=.1)
    plt.plot(r_x_mean, r_y_mean, linewidth=2, color=c)
    plt.fill_betweenx(r_y_mean, r_x_high, r_x_low, color=c, alpha=.1)
    
def graph_con_or_vis(reach_matrix):
    new_reach_matrix = [[],[],[],[]]
    for i in range(len(reach_matrix[0])):
        if reach_matrix[1][i][0] == 'v' and reach_matrix[3][i] == 'controlled':
            new_reach_matrix[0].append(reach_matrix[0][i, :])
            new_reach_matrix[1].append(reach_matrix[1][i])
            new_reach_matrix[2].append(reach_matrix[2][i])
            new_reach_matrix[3].append(reach_matrix[3][i])
    for i in range(3):
        new_reach_matrix[i] = np.asarray(new_reach_matrix[i])
    model, reaches, labels = sim_preprocessing(new_reach_matrix, 'vis')
    reaches = np.squeeze(reaches)
    print(reaches.shape)
    low_indices = [i[0] for i in np.argwhere\
                   (labels==0)]
    high_indices = [i[0] for i in np.argwhere\
                    (labels==1)]
    low_reaches = reaches[low_indices, :]
    high_reaches = reaches[high_indices, :]
    graph_from_matrix(low_reaches, 'r')
    graph_from_matrix(high_reaches, 'b')
    plt.show()

def pause_to_output(all_output):
    for ptp in all_output.keys():
        all_output[ptp]['max_speed'] = []
        all_output[ptp]['max_speed_t'] = []
        all_output[ptp]['min_speed'] = []
        all_output[ptp]['min_speed_t'] = []
        all_output[ptp]['upward'] = []
        data = get_data(get_raw_data(f'{ptp}.csv'))
        points = [line[12] for line in data]
        times = [line[13] for line in data]
        for i in range(len(points)):
            spec = speed_spec(points[i], times[i])
            all_output[ptp]['max_speed'].append(spec[0])
            all_output[ptp]['max_speed_t'].append(spec[1])
            all_output[ptp]['min_speed'].append(spec[2])
            all_output[ptp]['min_speed_t'].append(spec[3])
            if spec[2] < 1:
                all_output[ptp]['upward'].append(spec[4])
            else:
                all_output[ptp]['upward'].append(np.nan)
        
def close():
    con = sqlite3.connect('test_result.db')
    con.close()
    con.close()
    con.close()
    os.remove('test_result.db')
    
if __name__ == "__main__":
    FULLNAME = {'v': 'visual', 'c': 'conceptual', 'n': 'none'}
    my_db = 'test_result.db'
    input_ = 'stimuli_sim_new.csv'
    ptp_list = {2:'v',7:'c',9:'v',10:'c',11:'v',12:'c',13:'v',14:'c',16:'c',
                18:'v',19:'c',20:'v',21:'c',22:'v',23:'c',24:'v',25:'v',27:'c'}
    for ptp in ptp_list:
        ptp_list[ptp] = FULLNAME[ptp_list[ptp]]
    
    df = pd.DataFrame({})
        
    initiate(my_db, input_)
    all_output = {}
    populate(my_db, ptp_list)
    
    all_output = par_info(my_db, ptp_list, all_output)
    close()
    pause_to_output(all_output)
    for ptp in ptp_list.keys():
        # print(ptp)
        new_df = pd.DataFrame(all_output[ptp])
        df = df.append(new_df)
    df['speed'] = df['length'] / (df['end_time'] - df['start_time'])
    df['reach_time'] = df['end_time'] - df['start_time']
    df['any_sim'] = (df['conceptual_similarity'] + df['visual_similarity'])/2
    df['conceptual_bin'] = np.where(df['conceptual_similarity'] <= 0.5, 'low', 'high')
    df['visual_bin'] = np.where(df['visual_similarity'] <= 0.5, 'low', 'high')
    rating_d = {'v': 'visual', 'c': 'conceptual'}
    df['rating'] = np.vectorize(rating_d.get)(df['rating'])
    a = df.groupby('ptp')['speed'].median()
    a = a.rename('speed_median')
    df = pd.merge(df, a, on='ptp', how='inner')
    df['reach_type'] = np.where(df['speed'] >= df['speed_median'],
                                'automatic', 'controlled')
    df.reset_index(drop=True, inplace=True)
    df.to_csv('dataframe.csv')
    
    # all_reaches = interpolate_all_reaches(100, all_output)
    # reach_matrix = reach_matrix(all_reaches, df)
    # # PCA(reach_matrix)
    # # decoder_result = compare_decoder(reach_matrix, 'type')
    # # decoder_result = pd.read_csv('type_decoder_result.csv')
    # # decoder_mean = decoder_result.groupby('type').mean()
    # # decoder_std = decoder_result.groupby('type').std()
    # # order = ['rating and target', 'rating', 'target', 'congruent']
    # # decoder_group.loc[order].plot(yerr=decoder_result.groupby('type').std(), kind='bar')
    # plt.show()
    
    # # graph_by_style(reach_matrix, 'con_or_vis')
    # reaches = reach_matrix[0]
    # graph_con_or_vis(reach_matrix)
    
    
    # graph(df)
    
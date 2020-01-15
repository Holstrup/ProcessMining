import pandas as pd


def format_and_msdialog(path):
    df=pd.read_csv(path, sep='\t',header=None)


    # new data frame with split value columns
    features = df[1].str.split(" ", expand = True)
    features=features.rename(columns={0:'IUS', 1:'DS', 2:'?', 3:'D', 4:'What',
                                            5:'Where', 6:'When', 7:'Why', 8:'Who', 9:'How',
                                            10:'AP', 11:'NP', 12:'UL', 13:'ULU', 14:'ULSU', 15:'IS',
                                            16:'T', 17:'!', 18:'FB', 19:'SS(Neu)',
                                      20:'SS(Pos)', 21:'SS(Neg)', 22:'Lex(Pos)', 23:'Lex(Neg)'})
    df.drop(columns = [1], inplace=True)
    df=df.rename(columns={0:'Class'})



    training_set=pd.concat([df, features],axis=1)


    multiple_classes = training_set[training_set['Class'].str.contains("_")].index



    #df.loc[np.repeat(df.index.values, reps)]
    #print(multiple_classes['Class'].str.split("_", expand=True))
    for id,row in training_set.iterrows():
        if "_" in row['Class']:
            training_set=training_set.drop([id], axis=0)
            new_frame=pd.DataFrame()
            for each_class in row['Class'].split("_"):
                new_frame=new_frame.append(row.replace(row['Class'], each_class))
            training_set = training_set.append(new_frame)



    return training_set


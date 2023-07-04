import pandas as pd
import numpy as np
from sklearn import preprocessing
train_filepath = 'D:/IDdataset/NSL-KDD/KDDTrain+.txt'
test_filepath = 'D:/IDdataset/NSL-KDD/KDDTest+.txt'
destination_filepath = 'D:/IDdataset/processedNSL/concatenated.csv'

column_list = ['duration','protocol_type','service','flag','src_bytes',
              'dst_bytes','land','wrong_fragment','urgent','hot',
              'num_failed_logins', 'logged_in', 'num_compromised','root_shell','su_attempted',
              'num_root','num_file_creation','num_shells','num_access_files','num_outbound_cmds',
              'is_host_login','is_guest_login','count','srv_count','serror_rate',
              'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
              'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
              'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
              'dst_host_srv_rerror_rate','label','difficulty_level']


def one_hot(df):
    protocol_type_one_hot = pd.get_dummies(df["protocol_type"], prefix="protocol_type")
    df = df.drop("protocol_type", axis=1)
    df = df.join(protocol_type_one_hot)
    #
    service_one_hot = pd.get_dummies(df["service"], prefix="service")
    df = df.drop("service", axis=1)
    df = df.join(service_one_hot)
    #
    flag_one_hot = pd.get_dummies(df["flag"], prefix="flag")
    df = df.drop("flag", axis=1)
    df = df.join(flag_one_hot)
    # #
    # land_one_hot = pd.get_dummies(df["land"], prefix="land")
    # df = df.drop("land", axis=1)
    # df = df.join(land_one_hot)
    # #
    # logged_in_one_hot = pd.get_dummies(df["logged_in"], prefix="logged_in")
    # df = df.drop("logged_in", axis=1)
    # df = df.join(logged_in_one_hot)
    # #
    # # root_shell_one_hot = pd.get_dummies(df["root_shell"], prefix="root_shell")
    # # df = df.drop("root_shell", axis=1)
    # # df = df.join(root_shell_one_hot)
    # #
    # # su_attempted_one_hot = pd.get_dummies(df["su_attempted"], prefix="su_attempted")
    # # df = df.drop("su_attempted", axis=1)
    # # df = df.join(su_attempted_one_hot)
    # #
    # is_host_login_one_hot = pd.get_dummies(df["is_host_login"], prefix="is_host_login")
    # df = df.drop("is_host_login", axis=1)
    # df = df.join(is_host_login_one_hot)
    #
    # is_guest_login_one_hot = pd.get_dummies(df["is_guest_login"], prefix="is_guest_login")
    # df = df.drop("is_guest_login", axis=1)
    # df = df.join(is_guest_login_one_hot)

    # label_one_hot = pd.get_dummies(df["label"], prefix="label")
    # df = df.drop("label", axis=1)
    # df = df.join(label_one_hot)
    return df


def MinMax(train,test):
    columns_to_normalize = ['duration', 'src_bytes', 'dst_bytes',
                            'wrong_fragment', 'urgent', 'hot',
                            'num_failed_logins', 'num_compromised','root_shell',
                            'su_attempted','num_root',
                            'num_file_creation','num_shells', 'num_access_files',
                            'num_outbound_cmds','count', 'srv_count',
                            'serror_rate','srv_serror_rate', 'rerror_rate',
                            'srv_rerror_rate','same_srv_rate', 'diff_srv_rate',
                            'srv_diff_host_rate',
                            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                            'dst_host_srv_rerror_rate']
    min_values = train[columns_to_normalize].min()
    max_values = train[columns_to_normalize].max()

    train[columns_to_normalize] = (train[columns_to_normalize] - min_values) / (max_values - min_values)
    test[columns_to_normalize] = (test[columns_to_normalize] - min_values) / (max_values - min_values)
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    train.to_csv('D:/IDdataset/reprocess-NSL/train(MinMax).csv',header=True,index=False)
    test.to_csv('D:/IDdataset/reprocess-NSL/test(MinMax).csv', header=True, index=False)


if __name__ == '__main__':

    # df_train = pd.read_table(train_filepath,delimiter=',',names=column_list)
    # df_test = pd.read_table(test_filepath,delimiter=',',names=column_list)
    # df_concat = pd.concat([df_train,df_test])
    #
    # no_novelty_label = pd.read_csv('D:/IDdataset/no_novelty_test_label_origin.csv',header=0)
    # # print(no_novelty_label.value_counts())
    # Dos_list = ['apache2','back','land','mailbomb','neptune','pod','processtable','smurf','teardrop','udpstorm','worm']
    # Probe_list = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
    # U2R_list = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
    # R2L_list = ['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail','snmpgetattack',
    #             'snmpguess','spy','warezclient','warezmaster','xlock','xsnoop']
    #
    # for sub in Dos_list:
    #     no_novelty_label['label'].replace(sub,'dos',inplace=True)
    # for sub in Probe_list:
    #     no_novelty_label['label'].replace(sub,'probe',inplace=True)
    # for sub in U2R_list:
    #     no_novelty_label['label'].replace(sub,'u2r',inplace=True)
    # for sub in R2L_list:
    #     no_novelty_label['label'].replace(sub,'r2l', inplace=True)
    # no_novelty_label_oh = one_hot(no_novelty_label)
    # no_novelty_label_oh.reindex(columns=['label_normal','label_dos','label_probe','label_u2r','label_r2l'])
    # no_novelty_label.to_csv('D:/IDdataset/no_novelty_test_label_5class.csv',index=False)
    # print(df_concat)
    # df_concat.to_csv('D:/IDdataset/no_novelty_test_label.csv',index=False)

    # df_concat1 = pd.read_csv(destination_filepath,header=0)
    # # print(df_concat1)
    # df_onehot = one_hot(df_concat1)
    # df_onehot.fillna(0, inplace=True)
    # df_onehot.to_csv('D:/IDdataset/processedNSL/concat_onehot.csv',header=True,index=False)
    # data_in_order = df_onehot.reindex(columns=[
    #     'duration',
    #     'protocol_type_icmp','protocol_type_tcp','protocol_type_udp',
    #     'service_aol', 'service_auth', 'service_bgp', 'service_courier', 'service_csnet_ns', 'service_ctf', 'service_daytime',
    #     'service_discard', 'service_domain', 'service_domain_u', 'service_echo', 'service_eco_i', 'service_ecr_i', 'service_efs',
    #     'service_exec', 'service_finger', 'service_ftp', 'service_ftp_data', 'service_gopher', 'service_harvest','service_hostnames',
    #     'service_http', 'service_http_2784', 'service_http_443', 'service_http_8001', 'service_imap4', 'service_IRC','service_iso_tsap',
    #     'service_klogin', 'service_kshell', 'service_ldap', 'service_link', 'service_login', 'service_mtp','service_name',
    #     'service_netbios_dgm', 'service_netbios_ns', 'service_netbios_ssn', 'service_netstat', 'service_nnsp', 'service_nntp','service_ntp_u',
    #     'service_other', 'service_pm_dump', 'service_pop_2', 'service_pop_3', 'service_printer', 'service_private','service_red_i',
    #     'service_remote_job', 'service_rje', 'service_shell', 'service_smtp', 'service_sql_net', 'service_ssh','service_sunrpc',
    #     'service_supdup', 'service_systat', 'service_telnet', 'service_tftp_u', 'service_tim_i', 'service_time','service_urh_i',
    #     'service_urp_i', 'service_uucp', 'service_uucp_path', 'service_vmnet', 'service_whois', 'service_X11','service_Z39_50',
    #     'flag_OTH', 'flag_REJ', 'flag_RSTO', 'flag_RSTOS0', 'flag_RSTR',
    #     'flag_S0', 'flag_S1', 'flag_S2', 'flag_S3','flag_SF', 'flag_SH',
    #     'src_bytes', 'dst_bytes',
    #     'land_0', 'land_1',
    #     'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    #     'logged_in_0', 'logged_in_1',
    #     'num_compromised',
    #     'root_shell_0', 'root_shell_1',
    #     'su_attempted_0', 'su_attempted_1','su_attempted_2',
    #     'num_root', 'num_file_creation', 'num_shells',
    #     'num_access_files', 'num_outbound_cmds',
    #     'is_host_login_0', 'is_host_login_1',
    #     'is_guest_login_0', 'is_guest_login_1',
    #     'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    #     'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    #     'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    #     'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    #     'label_normal','label_dos','label_probe','label_u2r','label_r2l',
    #     'difficulty_level'
    # ])
    # df = pd.read_csv('D:/IDdataset/reprocess-NSL/no_novelty_test_encoded.csv',header=None)
    # # print(df[100])
    # for sub in ['apache2','mailbomb','processtable','udpstorm','worm',
    #             'mscan','saint',
    #             'ps','sqlattack','xterm',
    #             'httptunnel','named','sendmail','snmpgetattack','snmpguess','xlock','xsnoop']:
    #     indexNames = df[df[100] == sub].index
    #     df.drop(indexNames, inplace=True)
    # df.to_csv('D:/IDdataset/reprocess-NSL/no_novelty_test_encoded.csv',index=None,header=None)
    # df_sub_label = pd.read_csv('D:/IDdataset/reprocess-NSL/subclass_label.csv',header=0)
    # sub_label = one_hot(df_sub_label)
    # sub_label.to_csv('D:/IDdataset/reprocess-NSL/subclass_label_onehot.csv',header=True,index=None)
    # df_5class_label = pd.read_table('D:/IDdataset/NSL-KDD/KDDTrain+.txt', header=0)
    #
    # Dos_list = ['apache2','back','land','mailbomb','neptune','pod','processtable','smurf','teardrop','udpstorm','worm']
    # Probe_list = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
    # U2R_list = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
    # R2L_list = ['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail','snmpgetattack',
    #             'snmpguess','spy','warezclient','warezmaster','xlock','xsnoop']
    #
    # for sub in Dos_list:
    #     df_5class_label['label'].replace(sub,'1',inplace=True)
    # for sub in Probe_list:
    #     df_5class_label['label'].replace(sub,'2',inplace=True)
    # for sub in U2R_list:
    #     df_5class_label['label'].replace(sub,'3',inplace=True)
    # for sub in R2L_list:
    #     df_5class_label['label'].replace(sub,'4', inplace=True)
    #
    # # five_class_label = one_hot(df_5class_label)
    # five_class_label.to_csv('',header=True,index=None)
    feature = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
               "urgent", "hot",
               "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
               "num_file_creations", "num_shells",
               "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count",
               "serror_rate", "srv_serror_rate",
               "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
               "dst_host_count", "dst_host_srv_count",
               "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
               "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
               "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]

    train_df = pd.read_csv('D:/IDdataset/NSL-KDD/KDDTrain+.txt', names=feature)
    test_df = pd.read_csv('D:/IDdataset/NSL-KDD/KDDTest+.txt', names=feature)
    # test_data21 = pd.read_csv(test21, names=feature)
    concat_df = pd.concat([train_df, test_df], ignore_index=True)
    concat_df.drop(['difficulty'], axis=1, inplace=True)
    # concat_df.info()
    Dos_list = ['apache2', 'back', 'land', 'mailbomb', 'neptune', 'pod', 'processtable', 'smurf', 'teardrop',
                'udpstorm', 'worm']
    Probe_list = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
    U2R_list = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
    R2L_list = ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail',
                'snmpgetattack',
                'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop']
    for sub in Dos_list:
        concat_df['label'].replace(sub, 'dos', inplace=True)
    for sub in Probe_list:
        concat_df['label'].replace(sub, 'probe', inplace=True)
    for sub in U2R_list:
        concat_df['label'].replace(sub, 'u2r', inplace=True)
    for sub in R2L_list:
        concat_df['label'].replace(sub, 'r2l', inplace=True)
    label = pd.DataFrame(concat_df.label)
    print(label.value_counts())
    minmax_scaler = preprocessing.MinMaxScaler()

    def standardization(df, col):
        for i in col:
            arr = df[i]
            arr = np.array(arr)
            df[i] = minmax_scaler.fit_transform(arr.reshape(len(arr), 1))
        return df
    numeric_col = concat_df.select_dtypes(include='number').columns
    concat_df = standardization(concat_df, numeric_col)
    concat_df = pd.get_dummies(concat_df, columns=['protocol_type'], prefix="protocol_type")
    concat_df = pd.get_dummies(concat_df, columns=['service'], prefix="service")
    concat_df = pd.get_dummies(concat_df, columns=['flag'], prefix="flag")
    concat_df = pd.get_dummies(concat_df, columns=['label'], prefix="label")
    data_in_order = concat_df.reindex(columns=[
        'duration',
        'protocol_type_icmp','protocol_type_tcp','protocol_type_udp',
        'service_aol', 'service_auth', 'service_bgp', 'service_courier', 'service_csnet_ns', 'service_ctf', 'service_daytime',
        'service_discard', 'service_domain', 'service_domain_u', 'service_echo', 'service_eco_i', 'service_ecr_i', 'service_efs',
        'service_exec', 'service_finger', 'service_ftp', 'service_ftp_data', 'service_gopher', 'service_harvest','service_hostnames',
        'service_http', 'service_http_2784', 'service_http_443', 'service_http_8001', 'service_imap4', 'service_IRC','service_iso_tsap',
        'service_klogin', 'service_kshell', 'service_ldap', 'service_link', 'service_login', 'service_mtp','service_name',
        'service_netbios_dgm', 'service_netbios_ns', 'service_netbios_ssn', 'service_netstat', 'service_nnsp', 'service_nntp','service_ntp_u',
        'service_other', 'service_pm_dump', 'service_pop_2', 'service_pop_3', 'service_printer', 'service_private','service_red_i',
        'service_remote_job', 'service_rje', 'service_shell', 'service_smtp', 'service_sql_net', 'service_ssh','service_sunrpc',
        'service_supdup', 'service_systat', 'service_telnet', 'service_tftp_u', 'service_tim_i', 'service_time','service_urh_i',
        'service_urp_i', 'service_uucp', 'service_uucp_path', 'service_vmnet', 'service_whois', 'service_X11','service_Z39_50',
        'flag_OTH', 'flag_REJ', 'flag_RSTO', 'flag_RSTOS0', 'flag_RSTR',
        'flag_S0', 'flag_S1', 'flag_S2', 'flag_S3','flag_SF', 'flag_SH',
        'src_bytes', 'dst_bytes',
        'land','wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in','num_compromised','root_shell',
        'su_attempted', 'num_root', 'num_file_creation', 'num_shells',
        'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login',
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'label_normal','label_dos','label_probe','label_r2l','label_u2r'])
    data_in_order.fillna(0, inplace=True)
    data_in_order.to_csv('D:/IDdataset/feat122/concat.csv',header=True,index=False)
    # print(data_in_order)







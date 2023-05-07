import pandas as pd
import numpy as np

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
    # protocol_type_one_hot = pd.get_dummies(df["protocol_type"], prefix="protocol_type")
    # df = df.drop("protocol_type", axis=1)
    # df = df.join(protocol_type_one_hot)
    #
    # service_one_hot = pd.get_dummies(df["service"], prefix="service")
    # df = df.drop("service", axis=1)
    # df = df.join(service_one_hot)
    #
    # flag_one_hot = pd.get_dummies(df["flag"], prefix="flag")
    # df = df.drop("flag", axis=1)
    # df = df.join(flag_one_hot)
    #
    # land_one_hot = pd.get_dummies(df["land"], prefix="land")
    # df = df.drop("land", axis=1)
    # df = df.join(land_one_hot)
    #
    # logged_in_one_hot = pd.get_dummies(df["logged_in"], prefix="logged_in")
    # df = df.drop("logged_in", axis=1)
    # df = df.join(logged_in_one_hot)
    #
    # root_shell_one_hot = pd.get_dummies(df["root_shell"], prefix="root_shell")
    # df = df.drop("root_shell", axis=1)
    # df = df.join(root_shell_one_hot)
    #
    # su_attempted_one_hot = pd.get_dummies(df["su_attempted"], prefix="su_attempted")
    # df = df.drop("su_attempted", axis=1)
    # df = df.join(su_attempted_one_hot)
    #
    # is_host_login_one_hot = pd.get_dummies(df["is_host_login"], prefix="is_host_login")
    # df = df.drop("is_host_login", axis=1)
    # df = df.join(is_host_login_one_hot)
    #
    # is_guest_login_one_hot = pd.get_dummies(df["is_guest_login"], prefix="is_guest_login")
    # df = df.drop("is_guest_login", axis=1)
    # df = df.join(is_guest_login_one_hot)

    label_one_hot = pd.get_dummies(df["label"], prefix="label")
    df = df.drop("label", axis=1)
    df = df.join(label_one_hot)

    return df


def MinMax(train,test):
    columns_to_normalize = ['duration', 'src_bytes', 'dst_bytes',
                            'wrong_fragment', 'urgent', 'hot',
                            'num_failed_logins', 'num_compromised','num_root',
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
    train.to_csv('D:/IDdataset/processedNSL/train(MinMax).csv',header=True,index=False)
    test.to_csv('D:/IDdataset/processedNSL/test(MinMax).csv', header=True, index=False)


if __name__ == '__main__':
    # df_train = pd.read_table(train_filepath,delimiter=',',names=column_list)
    # df_test = pd.read_table(test_filepath,delimiter=',',names=column_list)
    # df_concat = pd.concat([df_train,df_test])
    #
    no_novelty_label = pd.read_csv('D:/IDdataset/no_novelty_test_label_origin.csv',header=0)
    # print(no_novelty_label.value_counts())
    Dos_list = ['apache2','back','land','mailbomb','neptune','pod','processtable','smurf','teardrop','udpstorm','worm']
    Probe_list = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
    U2R_list = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
    R2L_list = ['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail','snmpgetattack',
                'snmpguess','spy','warezclient','warezmaster','xlock','xsnoop']

    for sub in Dos_list:
        no_novelty_label['label'].replace(sub,'dos',inplace=True)
    for sub in Probe_list:
        no_novelty_label['label'].replace(sub,'probe',inplace=True)
    for sub in U2R_list:
        no_novelty_label['label'].replace(sub,'u2r',inplace=True)
    for sub in R2L_list:
        no_novelty_label['label'].replace(sub,'r2l', inplace=True)
    # no_novelty_label_oh = one_hot(no_novelty_label)
    # no_novelty_label_oh.reindex(columns=['label_normal','label_dos','label_probe','label_u2r','label_r2l'])
    no_novelty_label.to_csv('D:/IDdataset/no_novelty_test_label_5class.csv',index=False)
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
    # data_in_order.fillna(0, inplace=True)
    # data_in_order.to_csv('D:/IDdataset/processedNSL/concat_onehot_ordered.csv', header=True,index=False)
    #
    # df_train = pd.read_csv('D:/IDdataset/processedNSL/train(onehot).csv',header=0)
    # train = df_train.iloc[:,0:129]
    # # train = train.values
    #
    # df_test = pd.read_csv('D:/IDdataset/processedNSL/test(onehot).csv', header=0)
    # test = df_test.iloc[:, 0:129]
    # # test = test.values
    # # print(train.shape)
    # # print(test.shape)
    # # MinMax(train,test)
    # MinMax(train=train,test=test)



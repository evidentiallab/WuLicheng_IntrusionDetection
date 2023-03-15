import pandas as pd

# 读取CSV文件
data = pd.read_csv("../dataset/KDDCUP99/concatenated/concatenated-onehot.csv",header=0)
train = data.loc[0:494020]
test = data.loc[494021:]
# print(train)
# print(test)
# 提取需要标准化的列
columns_to_normalize = ['duration','src_bytes','dst_bytes',
                        'wrong_fragment','urgent','hot',
                        'num_failed_logins','num_compromised','root_shell',
                        'su_attempted','num_root','num_file_creation',
                        'num_shells','num_access_files','num_outbound_cmds',
                        'count','srv_count','serror_rate',
                        'srv_serror_rate',	'rerror_rate','srv_rerror_rate',
                        'same_srv_rate',	'diff_srv_rate','srv_diff_host_rate',
                        'dst_host_count',	'dst_host_srv_count','dst_host_same_srv_rate',
                        'dst_host_diff_srv_rate','dst_host_same_src_port_rate',	'dst_host_srv_diff_host_rate',
                        'dst_host_serror_rate','dst_host_srv_serror_rate',	'dst_host_rerror_rate',
                        'dst_host_srv_rerror_rate']
# 计算最大值和最小值
min_values = train[columns_to_normalize].min()
max_values = train[columns_to_normalize].max()
# 标准化数据
normalized_data = (data[columns_to_normalize] - min_values) / (max_values - min_values)
# 合并标准化后的数据
data_normalized = pd.concat([data.drop(columns_to_normalize, axis=1), normalized_data], axis=1)
data_in_order = data_normalized.reindex(columns=[
    'duration',
    'protocol_type_icmp','protocol_type_tcp','protocol_type_udp',
    'service_IRC',	'service_X11',	'service_Z39_50',	'service_auth',	'service_bgp',	'service_courier',	'service_csnet_ns',	'service_ctf',	'service_daytime',	'service_discard',	'service_domain',	'service_domain_u',	'service_echo',	'service_eco_i',	'service_ecr_i',	'service_efs',	'service_exec',	'service_finger',	'service_ftp',	'service_ftp_data',	'service_gopher',	'service_hostnames',	'service_http',	'service_http_443',	'service_icmp',	'service_imap4',	'service_iso_tsap',	'service_klogin',	'service_kshell',	'service_ldap',	'service_link',	'service_login',	'service_mtp',	'service_name',	'service_netbios_dgm',	'service_netbios_ns',	'service_netbios_ssn',	'service_netstat',	'service_nnsp',	'service_nntp',	'service_ntp_u',	'service_other',	'service_pm_dump',	'service_pop_2',	'service_pop_3',	'service_printer',	'service_private',	'service_red_i',	'service_remote_job',	'service_rje',	'service_shell',	'service_smtp',	'service_sql_net',	'service_ssh',	'service_sunrpc',	'service_supdup',	'service_systat',	'service_telnet',	'service_tftp_u',	'service_tim_i',	'service_time',	'service_urh_i',	'service_urp_i',	'service_uucp',	'service_uucp_path',	'service_vmnet',	'service_whois',
    'flag_OTH',	'flag_REJ',	'flag_RSTO',	'flag_RSTOS0',	'flag_RSTR',	'flag_S0',	'flag_S1',	'flag_S2',	'flag_S3',	'flag_SF',	'flag_SH',
    'src_bytes',	'dst_bytes',
    'land_0',	'land_1',
    'wrong_fragment','urgent','hot','num_failed_logins',
    'logged_in_0',	'logged_in_1',
    'num_compromised',	'root_shell',	'su_attempted',	'num_root',	'num_file_creation',	'num_shells',	'num_access_files',	'num_outbound_cmds',
    'is_host_login_0',	'is_host_login_1',
    'is_guest_login_0',	'is_guest_login_1',
    'count',	'srv_count',	'serror_rate',	'srv_serror_rate',	'rerror_rate',	'srv_rerror_rate',	'same_srv_rate',	'diff_srv_rate',	'srv_diff_host_rate',	'dst_host_count',	'dst_host_srv_count',	'dst_host_same_srv_rate',	'dst_host_diff_srv_rate',	'dst_host_same_src_port_rate',	'dst_host_srv_diff_host_rate',	'dst_host_serror_rate',	'dst_host_srv_serror_rate',	'dst_host_rerror_rate',	'dst_host_srv_rerror_rate',
])

# 保存标准化后的数据
data_in_order.to_csv("../dataset/KDDCUP99/concatenated/concatenated-onehot-MinMax.csv", header=True, index=False)

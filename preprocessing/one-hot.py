# coding:utf-8
# 离散型特征one-hot化处理

from numpy import argmax
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import csv
"""
back,buffer_overflow,ftp_write,guess_passwd,imap,ipsweep,land,loadmodule,multihop,neptune,nmap,normal,perl,phf,pod,portsweep,rootkit,satan,smurf,spy,teardrop,warezclient,warezmaster.
duration: continuous.
protocol_type: symbolic.
service: symbolic.
flag: symbolic.
src_bytes: continuous.
dst_bytes: continuous.
land: symbolic.
wrong_fragment: continuous.
urgent: continuous.
hot: continuous.
num_failed_logins: continuous.
logged_in: symbolic.
num_compromised: continuous.
root_shell: continuous.(potential symbolic)
su_attempted: continuous.(potential symbolic)
num_root: continuous.
num_file_creations: continuous.
num_shells: continuous.
num_access_files: continuous.
num_outbound_cmds: continuous.
is_host_login: symbolic.
is_guest_login: symbolic.
count: continuous.
srv_count: continuous.
serror_rate: continuous.
srv_serror_rate: continuous.
rerror_rate: continuous.
srv_rerror_rate: continuous.
same_srv_rate: continuous.
diff_srv_rate: continuous.
srv_diff_host_rate: continuous.
dst_host_count: continuous.
dst_host_srv_count: continuous.
dst_host_same_srv_rate: continuous.
dst_host_diff_srv_rate: continuous.
dst_host_same_src_port_rate: continuous.
dst_host_srv_diff_host_rate: continuous.
dst_host_serror_rate: continuous.
dst_host_srv_serror_rate: continuous.
dst_host_rerror_rate: continuous.
dst_host_srv_rerror_rate: continuous.
"""


def one_hot(df):
    protocol_type_one_hot = pd.get_dummies(df["protocol_type"],prefix="protocol_type")
    df = df.drop("protocol_type", axis=1)
    df = df.join(protocol_type_one_hot)

    service_one_hot = pd.get_dummies(df["service"],prefix="service")
    df = df.drop("service", axis=1)
    df = df.join(service_one_hot)

    flag_one_hot = pd.get_dummies(df["flag"],prefix="flag")
    df = df.drop("flag", axis=1)
    df = df.join(flag_one_hot)

    land_one_hot = pd.get_dummies(df["land"],prefix="land")
    df = df.drop("land", axis=1)
    df = df.join(land_one_hot)

    logged_in_one_hot = pd.get_dummies(df["logged_in"], prefix="logged_in")
    df = df.drop("logged_in", axis=1)
    df = df.join(logged_in_one_hot)

    # root_shell_one_hot = pd.get_dummies(df["root_shell"], prefix="root_shell")
    # df = df.drop("root_shell", axis=1)
    # df = df.join(root_shell_one_hot)
    #
    # su_attempted_one_hot = pd.get_dummies(df["su_attempted"], prefix="su_attempted")
    # df = df.drop("su_attempted", axis=1)
    # df = df.join(su_attempted_one_hot)

    is_host_login_one_hot = pd.get_dummies(df["is_host_login"], prefix="is_host_login")
    df = df.drop("is_host_login", axis=1)
    df = df.join(is_host_login_one_hot)

    is_guest_login_one_hot = pd.get_dummies(df["is_guest_login"], prefix="is_guest_login")
    df = df.drop("is_guest_login", axis=1)
    df = df.join(is_guest_login_one_hot)

    # label_one_hot = pd.get_dummies(df["label"], prefix="label")
    # df = df.drop("label", axis=1)
    # df = df.join(label_one_hot)
    return df


if __name__ == '__main__':
    df = pd.read_csv('../dataset/KDDCUP99/concatenated/concatenated.csv',header=0)
    df = one_hot(df)
    # df = df.reindex(columns=['duration','protocol_type_icmp	protocol_type_tcp','protocol_type_udp',
    #                          service_IRC	service_X11	service_Z39_50	service_auth
    #                          service_bgp	service_courier	service_csnet_ns
    #                          service_ctf	service_daytime	service_discard	service_domain
    #                          service_domain_u	service_echo	service_eco_i	service_ecr_i
    #                          service_efs	service_exec	service_finger	service_ftp	service_ftp_data
    #                          service_gopher	service_hostnames	service_http	service_http_443	service_imap4
    #                          service_iso_tsap	service_klogin	service_kshell	service_ldap	service_link
    #                          service_login	service_mtp	service_name	service_netbios_dgm	service_netbios_ns
    #                          service_netbios_ssn	service_netstat	service_nnsp	service_nntp	service_ntp_u
    #                          service_other	service_pm_dump	service_pop_2	service_pop_3	service_printer	service_private
    #                          service_red_i	service_remote_job	service_rje	service_shell	service_smtp	service_sql_net
    #                          service_ssh	service_sunrpc	service_supdup	service_systat	service_telnet	service_tftp_u
    #                          service_tim_i	service_time	service_urh_i	service_urp_i	service_uucp	service_uucp_path
    #                          service_vmnet	service_whois	flag_OTH	flag_REJ	flag_RSTO	flag_RSTOS0	flag_RSTR
    #                          flag_S0	flag_S1	flag_S2	flag_S3	flag_SF	flag_SH	src_bytes	dst_bytes	land_0	land_1
    #                          wrong_fragment	urgent	hot	num_failed_logins	logged_in_0	logged_in_1	num_compromised	root_shell_0
    #                          root_shell_1	su_attempted_0	su_attempted_1	su_attempted_2	num_root	num_file_creation
    #                          num_shells	num_access_files	num_outbound_cmds	is_host_login_0	is_guest_login_0
    #                          is_guest_login_1	count	srv_count	serror_rate	srv_serror_rate	rerror_rate	srv_rerror_rate
    #                          same_srv_rate	diff_srv_rate	srv_diff_host_rate	dst_host_count	dst_host_srv_count
    #                          dst_host_same_srv_rate	dst_host_diff_srv_rate	dst_host_same_src_port_rate	dst_host_srv_diff_host_rate	dst_host_serror_rate
    #                          dst_host_srv_serror_rate	dst_host_rerror_rate	dst_host_srv_rerror_rate])
    # print(df)
    dataset = df.to_csv('../dataset/KDDCUP99/concatenated/concatenated-onehot.csv',header=True,index=None)



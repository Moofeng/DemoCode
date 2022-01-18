'''
博客链接：https://blog.moofeng.cn/wei-xie-shou-lie/li-yong-rita-jian-ce-beacon-tong-xin
参考项目：https://github.com/activecm/rita
参考文章：https://posts.bluraven.io/implementing-rita-using-kql-8ccb0ee8eeae
'''

import pandas as pd
import numpy as np

# 预处理数据
timestamp = 'Time'
src_ip = 'Source'
dst_ip = 'Destination'
dst_port = 'Destination Port'
protocol = 'Protocol'
data_size = 'Length'
delimiter = ','
columns_to_filter = [timestamp, src_ip, dst_ip, dst_port, protocol, data_size]
columns_to_groupby = [src_ip, dst_ip, dst_port, protocol]
columns_to_display = ['score', 'tsScore', 'dsScore', src_ip, dst_ip, protocol, dst_port, 'tsSkewScore', 'tsMadmScore', 'tsConnCountScore', 'dsSkewScore', 'dsMadmScore', 'dsSmallnessScore']
http_df = pd.read_csv('covenant_http.csv', sep=delimiter)
http_df = http_df.loc[:, columns_to_filter]
http_df[timestamp] = pd.to_datetime(http_df[timestamp])
http_df = http_df.groupby(columns_to_groupby).agg(list)
http_df.reset_index(inplace=True)

# 过滤掉通信次数较少的数据包
http_df['conn_count'] = http_df[timestamp].apply(lambda x: len(x))
http_df = http_df.loc[http_df['conn_count'] > 20]
http_df[timestamp] = http_df[timestamp].apply(lambda x: sorted(x))

# 计算通信间隔和数据包大小差值
http_df['time_deltas'] = http_df[timestamp].apply(lambda x: pd.Series(x).diff().dt.seconds.dropna().tolist())
http_df['size_deltas'] = http_df[data_size].apply(lambda x: pd.Series(x).diff().dropna().tolist())

# 计算 tsSkew
http_df['tsLow'] = http_df['time_deltas'].apply(lambda x: np.percentile(np.array(x), 25))
http_df['tsMid'] = http_df['time_deltas'].apply(lambda x: np.percentile(np.array(x), 50))
http_df['tsHigh'] = http_df['time_deltas'].apply(lambda x: np.percentile(np.array(x), 75))
http_df['tsBowleyNum'] = http_df['tsLow'] + http_df['tsHigh'] - 2*http_df['tsMid']
http_df['tsBowleyDen'] = http_df['tsHigh'] - http_df['tsLow']
http_df['tsSkew'] = http_df[['tsLow','tsMid','tsHigh','tsBowleyNum','tsBowleyDen']].apply(
    lambda x: x['tsBowleyNum'] / x['tsBowleyDen'] if x['tsBowleyDen'] != 0 and x['tsMid'] != x['tsLow'] and x['tsMid'] != x['tsHigh'] else 0.0, axis=1
    )

# 计算 dsSkew
http_df['dsLow'] = http_df['size_deltas'].apply(lambda x: np.percentile(np.array(x), 25))
http_df['dsMid'] = http_df['size_deltas'].apply(lambda x: np.percentile(np.array(x), 50))
http_df['dsHigh'] = http_df['size_deltas'].apply(lambda x: np.percentile(np.array(x), 75))
http_df['dsBowleyNum'] = http_df['dsLow'] + http_df['dsHigh'] - 2*http_df['dsMid']
http_df['dsBowleyDen'] = http_df['dsHigh'] - http_df['dsLow']
http_df['dsSkew'] = http_df[['dsLow','dsMid','dsHigh','dsBowleyNum','dsBowleyDen']].apply(
    lambda x: x['dsBowleyNum'] / x['dsBowleyDen'] if x['dsBowleyDen'] != 0 and x['dsMid'] != x['dsLow'] and x['dsMid'] != x['dsHigh'] else 0.0, axis=1
    )

# 计算 tsSkewScore 和 dsSkewScore
http_df['tsSkewScore'] = 1.0 - abs(http_df['tsSkew'])
http_df['dsSkewScore'] = 1.0 - abs(http_df['dsSkew'])

# 计算 tsMadm 和 dsMadm
http_df['tsMadm'] = http_df['time_deltas'].apply(lambda x: np.median(np.absolute(np.array(x) - np.median(np.array(x)))))
http_df['dsMadm'] = http_df['size_deltas'].apply(lambda x: np.median(np.absolute(np.array(x) - np.median(np.array(x)))))

# 计算 tsMadmScore 和 dsMadmScore
http_df['tsMadmScore'] = 1.0 - http_df['tsMadm'] / 30.0
http_df['tsMadmScore'] = http_df['tsMadmScore'].apply(lambda x: 0 if x < 0 else x)
http_df['dsMadmScore'] = 1.0 - http_df['dsMadm'] / 32.0
http_df['dsMadmScore'] = http_df['dsMadmScore'].apply(lambda x: 0 if x < 0 else x)

# 计算 tsConnCountScore
http_df['tsConnDiv'] = http_df[timestamp].apply(lambda x: (x[-1].to_pydatetime() - x[0].to_pydatetime()).seconds / 90)
http_df['tsConnCountScore'] = (http_df['conn_count']) / http_df['tsConnDiv']
http_df['tsConnCountScore'] = http_df['tsConnCountScore'].apply(lambda x: 1.0 if x > 1.0 else x)

# 计算 dsSmallnessScore
http_df['dsSmallnessScore'] = http_df[data_size].apply(lambda x: 1- (np.argmax(np.bincount(x)) / 65535))
http_df['dsSmallnessScore'] = http_df['dsSmallnessScore'].apply(lambda x: 0 if x < 0 else x)

# 计算 score
http_df['tsScore'] = (((http_df['tsSkewScore'] + http_df['tsMadmScore'] + http_df['tsConnCountScore']) / 3.0) * 1000) / 1000
http_df['dsScore'] = (((http_df['dsSkewScore'] + http_df['dsMadmScore'] + http_df['dsSmallnessScore']) / 3.0) * 1000) / 1000
http_df['score'] = (http_df['tsScore'] + http_df['dsScore']) / 2.0

http_df.sort_values(by= 'score', ascending=False, inplace=True, ignore_index=True)
print(http_df[http_df['score']>0.8][columns_to_display])
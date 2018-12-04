FROM centos:6

LABEL Solomonfield <whg19961229@gmail.com>

WORKDIR /root


ENV HADOOP_VERSION=2.7.7
# install chinese support, openssh-server, oracle-jdk and wget 

RUN yum -y install kde-l10n-Chinese telnet && \
	yum -y install glibc-common &&\
    localedef -c -f UTF-8 -i zh_CN zh_CN.utf8 && \
    yum install -y openssh-server which openssh-clients wget && \
    ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
	cat /root/.ssh/id_rsa.pub > /root/.ssh/authorized_keys && \
    # ssh without key
	echo "    IdentityFile ~/.ssh/id_rsa" >> /etc/ssh/ssh_config && \
	echo "    StrictHostKeyChecking=no" >> /etc/ssh/ssh_config && \
	chmod 600 /root/.ssh/authorized_keys && \
    # install java
    curl -LO 'https://download.oracle.com/otn-pub/java/jdk/8u191-b12/2787e4a523244c269598db4e85c51e0c/jdk-8u191-linux-x64.rpm' -H 'Cookie: oraclelicense=accept-securebackup-cookie' && \
    rpm -i jdk-8u191-linux-x64.rpm && \
    rm -rf jdk-8u191-linux-x64.rpm && \
    # install hadoop 2.7.7
    wget https://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz && \
    tar zxvf hadoop-${HADOOP_VERSION}.tar.gz -C /usr/local/ && \
    mv /usr/local/hadoop-${HADOOP_VERSION} /usr/local/hadoop && \
    rm hadoop-${HADOOP_VERSION}.tar.gz

COPY config/* /tmp/
    # config files
RUN mv /tmp/hadoop-env.sh /usr/local/hadoop/etc/hadoop/hadoop-env.sh && \
    mv /tmp/hdfs-site.xml /usr/local/hadoop/etc/hadoop/hdfs-site.xml && \ 
    mv /tmp/core-site.xml /usr/local/hadoop/etc/hadoop/core-site.xml && \
    mv /tmp/mapred-site.xml /usr/local/hadoop/etc/hadoop/mapred-site.xml && \
    mv /tmp/yarn-site.xml /usr/local/hadoop/etc/hadoop/yarn-site.xml && \
    mv /tmp/slaves /usr/local/hadoop/etc/hadoop/slaves && \
    mv /tmp/hadoop-entrypoint.sh /root/hadoop-entrypoint.sh && \
    chmod +x /root/hadoop-entrypoint.sh && \
    chmod +x /usr/local/hadoop/sbin/start-dfs.sh && \
    chmod +x /usr/local/hadoop/sbin/start-yarn.sh && \
    mkdir -p ~/hdfs/namenode && \ 
    mkdir -p ~/hdfs/datanode && \
    mkdir /usr/local/hadoop/logs && \
    yum clean all && \
    rm -rf /var/cache/yum

# set environment variable
ENV LANG "zh_CN.UTF-8"
ENV LC_ALL "zh_CN.UTF-8"
ENV JAVA_HOME=/usr/java/jdk1.8.0_191-amd64
ENV HADOOP_HOME=/usr/local/hadoop 
ENV PATH=/usr/local/hadoop/bin:/usr/local/hadoop/sbin:$JAVA_HOME/bin:$PATH




# Hdfs ports
EXPOSE 50010 50020 50070 50075 50090 8020 9000 \
# Mapred ports
    10020 19888 \
#Yarn ports
    8030 8031 8032 8033 8040 8042 8088 \
#Other ports 
    49707 2122

ENTRYPOINT [ "sh", "-c", "/root/hadoop-entrypoint.sh; bash"]

#!/bin/bash
startHadoop(){
	$HADOOP_HOME/sbin/start-dfs.sh
    $HADOOP_HOME/sbin/start-yarn.sh
	$HADOOP_HOME/sbin/mr-jobhistory-daemon.sh start historyserver
}

main(){
	service sshd restart
	/usr/local/hadoop/bin/hdfs namenode -format
	sleep 5
	if [ ${ROLE} == "master" ]
	then
		startHadoop
		sleep 8
	fi
	sleep 5
	if [ ${ROLE} == "master" ]
	then
		hadoop fs -mkdir -p /spark/history
		/usr/local/spark/sbin/start-all.sh
		/usr/local/spark/sbin/start-history-server.sh
	fi
}

main

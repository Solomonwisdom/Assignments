package raft

//
// this is an outline of the API that raft must expose to
// the service (or tester). see comments below for
// each of these functions for more details.
//
// rf = Make(...)
//   create a new Raft server.
// rf.Start(command interface{}) (index, term, isleader)
//   start agreement on a new log entry
// rf.GetState() (term, isLeader)
//   ask a Raft for its current term, and whether it thinks it is leader
// ApplyMsg
//   each time a new entry is committed to the log, each Raft peer
//   should send an ApplyMsg to the service (or tester)
//   in the same server.
//

import (
    "bytes"
    "encoding/gob"
    "math/rand"
    "sync"
    "time"
)
import "labrpc"

// import "bytes"
// import "encoding/gob"


const (
    FOLLOWER     = "follower"
    CANDIDATE    = "candidate"
    LEADER       = "leader"
)

const (
    HeartbeatCycle  = time.Millisecond * 50
    ElectionMinTime = 150
    ElectionMaxTime = 300
)

//
// as each Raft peer becomes aware that successive log entries are
// committed, the peer should send an ApplyMsg to the service (or
// tester) on the same server, via the applyCh passed to Make().
//
type ApplyMsg struct {
    Index       int
    Command     interface{}
    UseSnapshot bool   // ignore for lab2; only used in lab3
    Snapshot    []byte // ignore for lab2; only used in lab3
}

//
// A Go object implementing a single Raft peer.
//
type Raft struct {
    mu        sync.Mutex
    peers     []*labrpc.ClientEnd
    persister *Persister
    me        int // index into peers[]

    // Your data here.
    // Look at the paper's Figure 2 for a description of what
    // state a Raft server must maintain.
    // Persistent state on all servers
    currentTerm int // latest term server has seen
    votedFor    int // candidateId that received vote in current term

    grantedVotesCount int // candidate's granted vote number
    // State
    state   string
    timer   *time.Timer

}

// return currentTerm and whether this server
// believes it is the leader.
func (rf *Raft) GetState() (int, bool) {

    var term int
    var isleader bool

    term = rf.currentTerm
    isleader = rf.state == LEADER
    return term, isleader
}

//
// save Raft's persistent state to stable storage,
// where it can later be retrieved after a crash and restart.
// see paper's Figure 2 for a description of what should be persistent.
//
func (rf *Raft) persist() {
    // Your code here.
    // Example:
    // w := new(bytes.Buffer)
    // e := gob.NewEncoder(w)
    // e.Encode(rf.xxx)
    // e.Encode(rf.yyy)
    // data := w.Bytes()
    // rf.persister.SaveRaftState(data)
    buf := new(bytes.Buffer)
    e := gob.NewEncoder(buf)
    e.Encode(rf.currentTerm)
    e.Encode(rf.votedFor)
    rf.persister.SaveRaftState(buf.Bytes())
}

//
// restore previously persisted state.
//
func (rf *Raft) readPersist(data []byte) {
    // Your code here.
    // Example:
    // r := bytes.NewBuffer(data)
    // d := gob.NewDecoder(r)
    // d.Decode(&rf.xxx)
    // d.Decode(&rf.yyy)
    if data != nil{
        r := bytes.NewBuffer(data)
        d := gob.NewDecoder(r)
        d.Decode(&rf.currentTerm)
        d.Decode(&rf.votedFor)
    }
}




//
// example RequestVote RPC arguments structure.
//
type RequestVoteArgs struct {
    Term         int  // candidate’s term
    CandidateId  int  // candidate requesting vote
}

//
// example RequestVote RPC reply structure.
//
type RequestVoteReply struct {
    Term         int  // currentTerm, for candidate to update itself
    VoteGranted  bool // true means candidate received vote
}

//
// example RequestVote RPC handler.
//
func (rf *Raft) RequestVote(args RequestVoteArgs, reply *RequestVoteReply) {
    rf.mu.Lock()
    defer rf.mu.Unlock()

    // Reply false if term < currentTerm
    if args.Term < rf.currentTerm {
        reply.VoteGranted = false
        reply.Term = rf.currentTerm
        return
    }

    if args.Term == rf.currentTerm {
        // If votedFor is null or candidateId, grant vote
        if rf.votedFor == -1 || rf.votedFor == args.CandidateId {
            rf.votedFor = args.CandidateId
            rf.persist()
            reply.VoteGranted = true
        }
        reply.Term = rf.currentTerm
    } else {
        // If RPC request or response contains term T > currentTerm:
        // set currentTerm = T, convert to follower
        rf.state = FOLLOWER
        rf.currentTerm = args.Term
        rf.votedFor = args.CandidateId
        reply.VoteGranted = true
        rf.persist()
        rf.resetTimer()
        reply.Term = args.Term
    }
}

//
// example code to send a RequestVote RPC to a server.
// server is the index of the target server in rf.peers[].
// expects RPC arguments in args.
// fills in *reply with RPC reply, so caller should
// pass &reply.
// the types of the args and reply passed to Call() must be
// the same as the types of the arguments declared in the
// handler function (including whether they are pointers).
//
// returns true if labrpc says the RPC was delivered.
//
// if you're having trouble getting RPC to work, check that you've
// capitalized all field names in structs passed over RPC, and
// that the caller passes the address of the reply struct with &, not
// the struct itself.
//
func (rf *Raft) sendRequestVote(server int, args RequestVoteArgs, reply *RequestVoteReply) bool {
    ok := rf.peers[server].Call("Raft.RequestVote", args, reply)
    return ok
}

//
// handle vote result
//
func (rf *Raft) handleVoteResult(reply RequestVoteReply) {
    rf.mu.Lock()
    defer rf.mu.Unlock()

    // reply from an old term is ignored
    if reply.Term < rf.currentTerm {
        return
    }

    // newer reply let peer convert to follower
    if reply.Term > rf.currentTerm {
        rf.currentTerm = reply.Term
        rf.state = FOLLOWER
        rf.votedFor = -1
        rf.resetTimer()
        return
    }

    // a candidate is granted a vote
    if rf.state == CANDIDATE && reply.VoteGranted {
        rf.grantedVotesCount++
        if rf.grantedVotesCount >= len(rf.peers)/2+1 {
            rf.state = LEADER
            // Upon election: send initial empty AppendEntries RPCs (heartbeat) to each server;
            // repeat during idle periods to prevent election timeouts
            rf.SendAppendEntriesToAllFollowers()
            rf.resetTimer()
        }
        return
    }
}

//
// example AppendEntries RPC arguments structure.
//
type AppendEntriesArgs struct{
    Term         int // leader’s term
    LeaderId     int // so follower can redirect clients
}

//
// example AppendEntries RPC reply structure.
//
type AppendEntriesReply struct {
    Term        int  // currentTerm, for leader to update itself
    Success     bool // true if follower contained entry matching prevLogIndex and prevLogTerm
}

//
// example AppendEntries RPC handler.
//
func (rf *Raft) AppendEntries(args AppendEntriesArgs, reply *AppendEntriesReply) {
    rf.mu.Lock()
    defer rf.mu.Unlock()

    // reply from an old term is ignored
    if args.Term < rf.currentTerm {
        reply.Success = false
        reply.Term = rf.currentTerm
    } else {
        rf.state = FOLLOWER
        rf.currentTerm = args.Term
        rf.votedFor = -1
        reply.Term = args.Term
        reply.Success = true
    }
    rf.persist()
    rf.resetTimer()
}

func (rf *Raft) SendAppendEntries(server int, args AppendEntriesArgs, reply *AppendEntriesReply) bool {
    ok := rf.peers[server].Call("Raft.AppendEntries", args, reply)
    return ok
}

//
// Handle AppendEntry result
//
func (rf *Raft) handleAppendEntriesResult(server int, reply AppendEntriesReply) {
    rf.mu.Lock()
    defer rf.mu.Unlock()

    if rf.state != LEADER {
        return
    }

    // Leader should convert to Follower
    if reply.Term > rf.currentTerm {
        rf.currentTerm = reply.Term
        rf.state = FOLLOWER
        rf.votedFor = -1
        rf.resetTimer()
        return
    }

    if !reply.Success {
        rf.SendAppendEntriesToAllFollowers()
    }
}

//
// send AppendEntries to all followers
//
func (rf *Raft) SendAppendEntriesToAllFollowers() {
    for i := 0; i < len(rf.peers); i++ {
        // except the leader
        if i == rf.me {
            continue
        }
        args := AppendEntriesArgs{
            Term: rf.currentTerm,
            LeaderId: rf.me,
        }
        go func(server int, args AppendEntriesArgs) {
            var reply AppendEntriesReply
            ok := rf.SendAppendEntries(server, args, &reply)
            if ok {
                rf.handleAppendEntriesResult(server, reply)
            }
        }(i, args)
    }
}

//
// the service using Raft (e.g. a k/v server) wants to start
// agreement on the next command to be appended to Raft's log. if this
// server isn't the leader, returns false. otherwise start the
// agreement and return immediately. there is no guarantee that this
// command will ever be committed to the Raft log, since the leader
// may fail or lose an election.
//
// the first return value is the index that the command will appear at
// if it's ever committed. the second return value is the current
// term. the third return value is true if this server believes it is
// the leader.
//
func (rf *Raft) Start(command interface{}) (int, int, bool) {
    index := -1
    term := -1
    isLeader := true


    return index, term, isLeader
}

//
// Upon timeout, this works
//
func (rf *Raft) handleTimer() {
    rf.mu.Lock()
    defer rf.mu.Unlock()

	// upon heartbeat timeout, it converts to candidate and sendRequestVote.
    if rf.state != LEADER {
        rf.state = CANDIDATE
        rf.currentTerm += 1
        rf.votedFor = rf.me
        // vote for itself
        rf.grantedVotesCount = 1
        rf.persist()
        args := RequestVoteArgs{
            Term:         rf.currentTerm,
            CandidateId:  rf.me,
        }
		// start a new election
        for server := 0; server < len(rf.peers); server++ {
            if server == rf.me {
                continue
            }

            go func(server int, args RequestVoteArgs) {
                var reply RequestVoteReply
                ok := rf.sendRequestVote(server, args, &reply)
                if ok {
                    rf.handleVoteResult(reply)
                }
            }(server, args)
        }
    } else {
        rf.SendAppendEntriesToAllFollowers()
    }
    rf.resetTimer()
}

//
// LEADER(HeartBeat):50ms
// FOLLOWER:150~300
//
func (rf *Raft) resetTimer() {
    if rf.timer == nil {
        rf.timer = time.NewTimer(time.Millisecond * 1000)
        go func() {
            for {
                <-rf.timer.C
                rf.handleTimer()
            }
        }()
    }
    newTimeout := HeartbeatCycle
	// make sure the election timeouts don't always fire at the same time
	if rf.state != LEADER {
        newTimeout = time.Millisecond * time.Duration(ElectionMinTime+rand.Intn(ElectionMaxTime-ElectionMinTime))
    }
    rf.timer.Reset(newTimeout)
}

//
// the tester calls Kill() when a Raft instance won't
// be needed again. you are not required to do anything
// in Kill(), but it might be convenient to (for example)
// turn off debug output from this instance.
//
func (rf *Raft) Kill() {
    // Your code here, if desired.
}

//
// the service or tester wants to create a Raft server. the ports
// of all the Raft servers (including this one) are in peers[]. this
// server's port is peers[me]. all the servers' peers[] arrays
// have the same order. persister is a place for this server to
// save its persistent state, and also initially holds the most
// recent saved state, if any. applyCh is a channel on which the
// tester or service expects Raft to send ApplyMsg messages.
// Make() must return quickly, so it should start goroutines
// for any long-running work.
//
func Make(peers []*labrpc.ClientEnd, me int,
    persister *Persister, applyCh chan ApplyMsg) *Raft {
    rf := &Raft{}
    rf.peers = peers
    rf.persister = persister
    rf.me = me

    // Your initialization code here.
    rf.currentTerm = 0
    rf.votedFor = -1

    rf.state = FOLLOWER
    // initialize from state persisted before a crash
    rf.readPersist(persister.ReadRaftState())
    rf.persist()
    rf.resetTimer()

    return rf
}

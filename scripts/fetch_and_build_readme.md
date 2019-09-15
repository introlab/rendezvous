1- Connect yourself to the jetson.

2- Go in the root directory:

    cd ./rendezvous
    
3- Activate the environment with:
    
    source ./env/bin/activate

4- You need to set a value for --branch. This is the branch name to sync on.

    python fetch_and_build.py --branch test-branch
    
5- You can optionally tell the script which commit on this branch to sync on (You need to enter the SHA1 hash of the commit). 

    python fetch_and_build.py --branch test-branch --commit 0983420183401830912213
    
6- Look at the logs to see if there is an error. If everything is OK, there will be a message at the end of the execution that tells you that everythings fine.

7- If an exception is thrown during the execution a rollback will be execute to comeback to the last state.

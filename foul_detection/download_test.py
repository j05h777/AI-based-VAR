from SoccerNet.Downloader import SoccerNetDownloader as SNdl

mySNdl = SNdl(LocalDirectory="D:\joshb\VAR_code\sn-mvfoul")
mySNdl.downloadDataTask(
    task="mvfouls",
    split=["test"], 
    version="720p", 
    password="s0cc3rn3t"
)

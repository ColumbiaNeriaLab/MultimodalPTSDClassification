import pickle
import os

class Log:
    '''
    A general logging utility
    
    attributes:
        log_name : str
            The filename for the log
        log_path : str
            The path for the log file
        curr_str : str
            The string being tracked currently
        total_str : str
            The total log file string
        should_log : bool
            Whether to log currently
        overwrite : bool
            Whether to overwrite the log file if it exists
    
    methods:
        enable()
            Enables logging
        disable()
            Disables logging
        clear()
            Clears the log entirely
        set_name(name)
            Sets the name of the log file
        set_path(path)
            Sets the path of the log file
        get_log()
            Returns the entire log
        print_log
            Prints the entire log
        save
            Saves the log to 'log_path/log_name'
        print_and_log
            Prints the specified arguments and then adds them
            as a string to the log
    '''
        
        
    def __init__(self, log_name='log.txt', log_path = '.', should_log=True, overwrite=True):
        '''
        params:
            log_name: str (default = 'log.txt')
                The name of the log file
            log_path: str (default = '.')
                The path of the log file
            should_log: bool (default = True)
                Whether to log currently
            overwrite: bool (default = True)
                Whether to overwrite the log file if it exists    
        '''
        
        self.log_name = log_name
        self.log_path = log_path
        self.curr_str = ""
        self.total_str = ""
        self.should_log = should_log
        self.overwrite = overwrite
        
    def enable(self):
        '''
        Enables logging
        '''
        
        self.should_log = True
        
    def disable(self):
        '''
        Disables logging
        '''
        
        self.should_log = False
        
    def clear(self):
        '''
        Clears the log entirely
        '''
        
        self.curr_str = ""
        self.total_str = ""
        
    def set_name(self, name):
        '''
        Sets the name of the log file
        
        params:
            name: str
                The new log file name
        '''
        
        self.log_name = name
        
    def set_path(self, path):
        '''
        Sets the path of the log file
        
        params:
            path: str
                The new log file path
        '''
        
        self.log_path = path
        
    def get_log(self):
        '''
        Returns the entire log
        
        return:
            str
                The entire log as a string
        '''
        
        return self.total_str
        
    def print_log(self):
        '''
        Prints the entire log
        '''
        
        print(self.total_str)
        
    def save(self):
        '''
        Saves the log to 'log_path/log_name'
        If self.overwrite, then the current log file is overwritten
        Otherwise, the current log file is appended to
        '''
        
        if self.should_log:
            fpath = os.path.join(self.log_path, self.log_name)
            if self.overwrite:
                with open(fpath, 'w+') as logfile:
                    logfile.write(self.total_str)
            else:
                with open(fpath, 'a') as logfile:
                    logfile.write(self.curr_str)
                self.curr_str = ""
                
    def print_and_log(self, *args, **kwargs):
        '''
        Prints and adds the current arguments to the log as a string
        If self.should_log, then the arguments are also logged
        Otherwise, they are solely printed
        '''
        if self.should_log:
            new_str= ""
            for s in args:
                new_str += str(s)
            new_str += '\n'
            self.curr_str += new_str
            self.total_str += new_str
            print(*args, **kwargs)
        else:
            print(*args, **kwargs)
            return None
            

class Stats:
    '''
    A general utility for tracking stats
    
    attributes:
        filename: str
            The name of the stats file
        path:
            The path of the stats file
        stats:
            The dictionary holding the stats
            
    methods:
        clear_stat(statname)
            Clears the stat information for the stat of the given name
        delete_stat(statname)
            Deletes the stat completely
        track_stat(statname, key, val)
            Tracks the stat of the given name with a given key value pair
        set_name(name)
            Sets the name of the stat file
        set_path(path)
            Sets the path of the stat file
        save()
            Saves the stats to 'path/filename'
            
    '''
    
    def __init__(self, filename='stats', path='.', load=True):
        '''
        params:
            filename: str (default = 'stats')
                The name of the stats file
            path: str (default = '.')
                The path of the stats file
            load: bool (default = True)
                Whether to load the stats file
        '''
        
        # Want to ensure stats files have .stt extension
        fname, fext = os.path.splitext(filename)
        if fext != ".stt" and fext != "":
            raise Exception("Filename '{}' does not have correct extension ('{}' instead of '.stt')".format(filename, fext))
        if fext != ".stt":
            self.filename = fname + ".stt"
            
        self.path = path
        
        filepath = os.path.join(self.path, self.filename)
        
        if load and os.path.exists(filepath):
            with open(filepath, 'rb') as pickle_file:
                self.stats = pickle.load(pickle_file)
        else:
            if load:
                print("Warning: You are trying to load a stats file '{}' that doesn't exist.".format(os.path.abspath(filepath))) 
            self.stats = {}
            
    def clear_stat(self, statname):
        '''
        Clears the stat information for the stat of the given name
        
        params:
            statname: str
                The name of the stat to be cleared
        '''
        
        if statname in self.stats:
            self.stats[statname] = {}
            
    def delete_stat(self, statname):
        '''
        Deletes the stat completely
        
        params:
            statname: str
                The name of the stat to be deleted
        '''
        
        del stats[statname]
        
    def track_stat(self, statname, key, val):
        '''
        Tracks the stat of the given name with a given key value pair
        
        params:
            statname
                The name of the stat to track
            key
                The current key for the given stat
            val
                The value at that current key
        '''
        
        if statname not in self.stats:
            self.stats[statname] = {}
        self.stats[statname][key] = val
    
    def set_name(self, name):
        '''
        Sets the name of the stats file
        
        params:
            name: str
                The new stats file name
        '''
        
        self.filename = name
        
    def set_path(self, path):
        '''
        Sets the path of the stats file
        
        params:
            path: str
                The new stats file path
        '''
        
        self.path = path
        
    def save(self):
        '''
        Saves the stats to 'path/filename'
        '''
        
        filepath = os.path.join(self.path, self.filename)
        with open(filepath, 'wb') as pickle_file:
            pickle.dump(self.stats, pickle_file)
            
    def __getitem__(self, key):
        
        return self.stats[key]
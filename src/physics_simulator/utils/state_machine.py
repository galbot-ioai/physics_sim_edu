class SimpleStateMachine:
    def __init__(self, max_states=-1):
        """
        Initialize the state machine
        
        Args:
            max_states (int): Maximum number of states, -1 means unlimited
        """
        self.state_idx = 0
        self.new_state = True
        self.max_state_cnt = max_states
        self.state_names = {}  # Store state names
        self.state_callbacks = {}  # Store state callback functions
        
    def next(self):
        """
        Switch to the next state
        
        Returns:
            bool: True if successfully switched to next state, False otherwise
        """
        if self.max_state_cnt == -1 or self.state_idx < self.max_state_cnt:
            self.new_state = True
            self.state_idx += 1
            return True
        return False

    def trigger(self):
        """
        Check if this is a new state
        
        Returns:
            bool: True if this is a new state, False otherwise
        """
        if self.new_state:
            self.new_state = False
            return True
        return False
    
    def reset(self):
        """Reset the state machine to initial state"""
        self.state_idx = 0
        self.new_state = True
        
    def add_state(self, state_idx, name, callback=None):
        """
        Add a state with its callback function
        
        Args:
            state_idx (int): State index
            name (str): State name
            callback (callable, optional): State callback function
        """
        self.state_names[state_idx] = name
        if callback is not None:
            self.state_callbacks[state_idx] = callback
            
    def get_state_name(self):
        """
        Get the name of current state
        
        Returns:
            str: Name of current state, None if not defined
        """
        return self.state_names.get(self.state_idx)
        
    def execute_current_state(self, *args, **kwargs):
        """
        Execute the callback function of current state
        
        Args:
            *args: Arguments passed to the callback function
            **kwargs: Keyword arguments passed to the callback function
            
        Returns:
            Any: Return value of the callback function, None if no callback function defined
        """
        callback = self.state_callbacks.get(self.state_idx)
        if callback is not None:
            return callback(*args, **kwargs)
        return None
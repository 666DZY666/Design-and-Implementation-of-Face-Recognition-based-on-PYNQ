"""
The MovidiusNCS class simplifies interactions with the NCS device(s)
"""

from mvnc import mvncapi as mvnc


class MovidiusNCS:

    def __init__(self):
        # configuration NCS
        mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
        self.devices = mvnc.EnumerateDevices()
        if len(self.devices) == 0:
            # No NCS devices found
            print('No NCS devices found')
            self.device = None
        else:
            # Try connecting to the first NCS device found
            self.device = mvnc.Device(self.devices[0])
            self.device.OpenDevice()
            self.opt = self.device.GetDeviceOption(mvnc.DeviceOption.OPTIMISATION_LIST)
            
    '''
    load_graph: returns the graph object when provided the graph filename
    '''
    def load_graph(self,graph_filename):
        # If connected to an NCS device, then load the graph file
        if self.device:
            # load blob
            with open(graph_filename, mode='rb') as f:
                blob = f.read()
            self.graph = self.device.AllocateGraph(blob)
            self.graph.SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
            self.iterations = self.graph.GetGraphOption(mvnc.GraphOption.ITERATIONS)
            return(True)
        else:
            print('Not connected to NCS device')
            return(False)
        
    '''
    close: deallocates loaded graph and closes the connected NCS
    '''
    def close(self):
        # Deallocate the loaded graph
        self.graph.DeallocateGraph()
        # Close the Movidius NCS device
        self.device.CloseDevice()        


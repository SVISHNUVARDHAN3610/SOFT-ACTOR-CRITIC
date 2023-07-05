from main import Main
import os
main_path = "data\weights"
paths = []
paths.append(os.path.join(main_path,"policy.pth"))
paths.append(os.path.join(main_path,"q_value.pth"))
paths.append(os.path.join(main_path,"value.pth"))

if __name__ =="__main__":
    main = Main(210,6,4480,10000000,350,32,10000,0.99,0.95,False,paths)
    main.train()
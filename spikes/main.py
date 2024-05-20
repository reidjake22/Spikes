# the main module
if __name__ == '__main__':
    # this code is only run if the script is called from the command line
    print("treated as script")
    from brian2 import *
    import train_projects
    train_projects.project_1()
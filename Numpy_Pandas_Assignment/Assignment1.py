#Q1
import re
def names():
    simple_string = """Amy is 5 years old, and her sister Mary is 2 years old. 
    Ruth and Peter, their parents, have 3 kids."""

    answer=re.findall('[A-Z][a-z]{1,100}',simple_string)
    return answer
    #raise NotImplementedError()

#Q2
def grades():
    with open ("assets/grades.txt", "r") as file:
        grades = file.read()
    grades=grades.split('\n')
    answer=[]
    for grade in grades:
        name=re.findall('[\w]{1,100} [\w]{1,100}: B',grade)
        if len(name)>0:
            name=name[0].split(':')[0]
            answer.append(name)
    return answer

#Q3
import re
def logs():
    with open("assets/logdata.txt", "r") as file:
        logdata = file.read()
    
    # YOUR CODE HERE
    logdata=logdata.split('\n')
    dicts=[]
    for log in logdata:
        try:
            host = re.findall('^[\w]{1,3}.[\w]{1,3}.[\w]{1,3}.[\w]{1,3}',log)[0]
        except:
            continue
        username = re.findall('- [\w]{1,100} ',log)
        if len(username)>0:
            username=username[0].strip('-').strip(' ')
        else:
            username='-'
        try:
            time = re.findall('[\w]*/[\w]*/[\w]*:[\w]*:[\w]*:[\w]* -[\w]*',log)[0]
        except:
            continue
        request = log.split('"')
        request=request[1]
        dic = {'host':host,'user_name':username,'time':time,'request':request}
        dicts.append(dic)
    return dicts


def main():
    print(len(grades()))

main()

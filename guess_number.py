import random

answer = random.randint(1, 100)
counter = 0
while True:
    counter += 1
    number = int(input('请输入: '))
    if number < answer:
        print('大一点')
    elif number > answer:
        print('小一点')
    else:
        print('恭喜你猜对了!')
        break
print('你总共猜了%d次' % counter)
if counter > 7:
    print('你的智商余额明显不足')
elif counter <= 2:
    print('还等什么？ 快去买彩票！')
elif counter <= 3:
    print('OMG！你是机器人吗！？？？')
elif counter <= 5:
    print('你太聪明啦！我佩服你！')

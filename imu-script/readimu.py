

import csv  


def main():

    arrayX = []
    arrayY = []
    arrayT = []
    numberOfFilesToRead = 26
    firstTime = 0 
    count = 0

    file2 = open("myfile.txt","w")

    file2.write("[")

    for i in range(numberOfFilesToRead):
        # Using readlines()
        filename = 'catalog_' + str(i) + '.catalog'
        file1 = open(filename, 'r')
        Lines = file1.readlines()
        # Strips the newline character
        for line in Lines:
            # print("Line{}: {}".format(count, line))
            # result = line.find('imu/acl_x')
            result = line.find('imu/gyr_x')

            wordx = line[result+12: result+33]
            x3 =wordx.replace('"','')
            x4 = x3.replace("i","")
            x5 = x4.replace(",","")
            x6 = x5.replace("m","")
            x7 = x6.replace("u","")
            x8 = x7.strip()
            arrayX.append(x8)
            # print("Line{}: {}".format(count, x6))


            # print("Line{}: {}".format(count, line))
            # result = line.find('imu/acl_y')
            result = line.find('imu/gyr_y')

            wordy = line[result+12: result+33]
            y3 =wordy.replace('"','')
            y4 =y3.replace("i","")
            y5 = y4.replace(",","")
            y6 = y5.replace("m","")
            y7 = y6.replace("m","")
            y8 = y7.strip()
            arrayY.append(y8)
            # print("Line{}: {}".format(count, y6))

            # print("Line{}: {}".format(count, line))
            result = line.find('timestamp_ms')
            wordt = line[result+15: result+28]
            t3 =wordt.replace('"','')
            t4 =t3.replace("i","")
            t5 = t4.replace(",","")
            t6 = t5.strip()
            arrayT.append(t6)
            # print("Line{}: {}".format(count, t6))

            # print(t6)
            # print(count)

            L = x8 + " " + y8 + " " + str(0) + ";\n"
            file2.writelines(L)

            if (count == 0):
                firstTime = t6
            else: 
                totalsec = ((int(t6) - int(firstTime))/1000)
                print(str(totalsec))
            
            count += 1


    # print(arrayX)
    # print(arrayY)
    # print(arrayT)
    data = []



    # print (file1.read())

    
    # \n is placed to indicate EOL (End of Line)    
    file1.close() #to change file access modes


    # with open('catalogdata.csv', 'w') as f:
    #     writer = csv.writer(f)

    #     # write the header
    #     contercsv = 0
    #     for x in arrayT:
    #         data = [arrayX[contercsv], arrayY[contercsv], 0]
    #         contercsv = contercsv + 1
    #         # write the data
    #         writer.writerow(data)
    

if __name__ == "__main__":
    main()

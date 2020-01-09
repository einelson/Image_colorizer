import pickle
# ask user to read or write data
sw= input("Write (W), read (R) or edit (E) data: ")

if sw=="W" or sw=="w":
   # create save data
   data=input("Please put message to save: ")
   # get filename
   filename='picture_colorization/saved_files/' + input("Please enter a filename:") + ".sav"
   # save file
   try:
      pickle.dump(data, open(filename, 'wb'))
   except:
      print("Unable to save message")
elif sw=="R" or sw=="r":
   # get filename
   filename='picture_colorization/saved_files/' + input("Please enter a filename:") + ".sav"
   # load model
   try:
      data=pickle.load(open(filename, 'rb'))
      print(data)
   except:
      print("Unable to load message")
elif sw=="E" or sw=="e":
   # get filename
   filename='picture_colorization/saved_files/' + input("Please enter a filename:") + ".sav"
   # load model
   try:
      data=pickle.load(open(filename, 'rb'))
      print("Current data: " + data)
   except:
      print("Unable to load message")
   # enter new data
   data=input("Enter new data: ")
   try:
      pickle.dump(data, open(filename, 'wb'))
   except:
      print("Unable to save message")
else:
   print("Input not understood")
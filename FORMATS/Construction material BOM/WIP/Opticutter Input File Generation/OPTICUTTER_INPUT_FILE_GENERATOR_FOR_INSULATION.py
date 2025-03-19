import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import math
from time import sleep
from fractions import Fraction
from datetime import datetime

# Set the color scheme
bg_color = "#f8f8f8"  # Light background color
fg_color = "#333333"  # Dark text color
entry_bg_color = "#ffffff"  # White input field background
button_color = "#333333"  # Dark button color
button_text_color = "#ffffff"  # White text on the button

def getLabels(values):
    maxTextLength = len(max(values, key=len))
    windowWidth = maxTextLength*10 + 150
    if windowWidth < 350: windowWidth = 350
    windowHeight = len(values) * 30 + 150
    # print("Window Dimensions: ", windowWidth, "x", windowHeight)
    
    # Create a new window for user input
    popup = tk.Toplevel(root)
    popup.title("Opticutter Input Generator")
    project_name_label.pack(pady=5, padx=20, fill='x')
    popup.geometry(str(windowWidth) + "x" + str(windowHeight))

    # Configure the popup window
    popup.configure(bg=bg_color)

    # Create a scrollable frame
    canvas = tk.Canvas(popup, bg=bg_color)
    scrollbar = ttk.Scrollbar(popup, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg=bg_color)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Pack the canvas and scrollbar
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Create a title label
    popupTitle = tk.Label(
        scrollable_frame,
        text="Enter Labels for each part",
        font=("Helvetica", 14, "bold"),
        bg=bg_color,
        fg=fg_color
    )

    popupTitle.pack(side=tk.TOP, pady=10)

    labels = []

    # Function to validate label entries
    def validateLabels(P):
        if len(P) == 0:
            # empty Entry is ok
            return True
        elif len(P) <= 3:
            # Entry with 1 digit is ok
            return True
        else:
            # Anything else, reject it
            return False
        

    # Function to handle submission
    def submit(labels):

        for entry in entries:
            labels.append(entry.get())

        if "" in labels[-len(values):] or labels == []:
            # labels = []
            tk.messagebox.showerror("Error", "Please enter labels for all parts.")
        elif len(labels[-len(values):]) != len(set(labels[-len(values):])):
            # labels = []
            tk.messagebox.showerror("Error", "Labels should be unique values.")
        else:
            # Close the popup window
            popup.destroy()

    # Create a label and entry for each value in the list
    entries = []
    for value in values:
        frame = tk.Frame(scrollable_frame, bg=bg_color)
        frame.pack(pady=5, padx=20, fill='x')

        label = tk.Label(
            frame,
            text=value,
            bg=bg_color,
            fg=fg_color,
            anchor='w',
            width=maxTextLength
        )
        label.pack(side=tk.LEFT)

        vcmd = (popup.register(validateLabels), '%P')
        entry = tk.Entry(
            frame,
            validate="key",
            validatecommand=vcmd,
            bg=entry_bg_color,
            fg=fg_color,
            insertbackground=fg_color
        )
        entry.pack(side=tk.LEFT, padx=10, ipadx=10, fill='x')
        entries.append(entry)

    # Submit button
    submit_button = tk.Button(
        scrollable_frame,
        text="Submit",
        command=lambda: submit(labels),
        bg=button_color,
        fg=button_text_color,
        activebackground=button_text_color,
        activeforeground=button_color
    )
    submit_button.pack(pady=15, ipadx=10, ipady=5)
    
    root.withdraw()

    # Handle window close event
    def on_close():
        if tk.messagebox.askyesno("Exit", "Do you want to exit the tool?"):
            popup.destroy()
            root.destroy()
            exit()
    
    popup.protocol("WM_DELETE_WINDOW", lambda: on_close())

    root.wait_window(popup)

    return labels[-len(values):]



# creates an extended DataFrame (extendedDF) by iterating over unique parts and labels, 
# filtering the original DataFrame (df) for each part, adding a label column, and concatenating the results.
def createExtendedDF(df, uniqueParts, labels):
    extendedDF = pd.DataFrame()
    for i in range(len(uniqueParts)):

        # Filter the Dataframe for each unique part
        # THE PROBLEM IS SOMETIMES THE PART NUMBER IS A STRING WITH OR WITHOUT "-" AND SOMETIMES IT IS AN INT (1729520)
        # SO INITIALLY FILTERING THE DB BY CONSIDERING THE PART NUMBER AS A STRING
        # IF THE FILTERED DF TURNED OUT TO BE EMPTY THEN FILTER THE DB BY CONSIDERING THE PART NUMBER AS AN INT
        filteredDF = df.loc[df['PART'] == "-".join(uniqueParts[i].split("-")[:-1])].copy()
        if filteredDF.empty:
            filteredDF = df.loc[df['PART'] == int(uniqueParts[i].split("-")[0])].copy()

        filteredDF['LABEL'] = labels[i]
        extendedDF = pd.concat([extendedDF, filteredDF], ignore_index=True)
    
    extendedDF.reset_index(drop=True, inplace=True)

    return extendedDF


def matchPartsAndRawMaterials(partNumbers, itemNumbers, qty):
    partList = []
    rawMaterialList = []
    uniqueParts = []
    currentPart = ""
    for i in range(len(partNumbers)):
        if math.isnan(itemNumbers[i]):
            currentPart = partNumbers[i]
            # uniqueParts.extend([currentPart] * int(qty[i]))
            for j in range(int(qty[i])):
                uniqueParts.append(str(currentPart)+"-"+str(j+1))
            
        else:
            partList.append(currentPart)
            rawMaterialList.append(partNumbers[i].strip())

    return partList, rawMaterialList, uniqueParts

def create_part_label(parts, qty, item):
    if ',' in parts:
        part_list = parts.split(", ")
        qty_list = qty.split(", ")
        item_list = item.split(", ")

        # THE LABEL COLUMN IN THE OPTICUTTER WEBSITE ONLY PROCESSES 30 CHARACTERS.
        # BY THIS METHOD THE NUMBER OF CHARACTERS ARE CONSTANTLY GETTING OVER 30.
        # return ", ".join([f"{part}({item})x{qty}" for part, qty, item in zip(part_list, qty_list, item_list)])
        
        return ",".join([f"{part}#{item}x{qty}" for part, qty, item in zip(part_list, qty_list, item_list)])
    else:
        # THE LABEL COLUMN IN THE OPTICUTTER WEBSITE ONLY PROCESSES 30 CHARACTERS.
        # BY THIS METHOD THE NUMBER OF CHARACTERS ARE CONSTANTLY GETTING OVER 30.
        # return f"{parts}({item})x{qty}"
        
        return f"{parts}#{item}x{qty}"
    
def remove_duplicates(dimension):
    if ',' in dimension:
        dimension_list = list(dict.fromkeys(dimension.split(", ")))
        if len(dimension_list) == 1:
            return dimension_list[0]
        else:
            return dimension
    else:
        return dimension
    

def split_labels(df, label_col='Label', max_length=30):
    new_rows = []
    for _, row in df.iterrows():
        labels = row[label_col].split(',')
        current_row = row.copy()
        current_label = []
        
        for label in labels:
            if len(','.join(current_label + [label])) <= max_length:
                current_label.append(label)
            else:
                current_row[label_col] = ','.join(current_label)

                curQty = [int(x.split('x')[-1]) for x in current_label]
                curQty = sum(curQty)
                current_row["Quantity"] = curQty

                new_rows.append(current_row)
                current_row = row.copy()
                current_label = [label]
        
        # Add the remaining labels in the last row
        if current_label:
            current_row[label_col] = ','.join(current_label)

            curQty = [int(x.split('x')[-1]) for x in current_label]
            curQty = sum(curQty)
            current_row["Quantity"] = curQty
            
            new_rows.append(current_row)
    
    return pd.DataFrame(new_rows)

    
def prepare_stock_data(df, stockName):
    filteredStock = df[df['OPTIONS'] == stockName]

    if len(filteredStock) > 1:
        print(f"Multiple stock data for {stockName} found. Expecting Only one stock row. Exiting program.")
        tk.messagebox.showerror("Error", f"Multiple stock data for {stockName} found. Expecting Only one stock row. Exiting program.")
        exit()

    elif len(filteredStock) == 0:
        print(f"No stock data for {stockName} found. Exiting program.")
        tk.messagebox.showerror("Error", f"No stock data for {stockName} found. Exiting program.")
        exit()

    # print(filteredStock["Width_in"].values[0], filteredStock["Height_in"].values[0])
    stockDF = pd.DataFrame()
    stockDF = pd.DataFrame(columns=["@", "Length", "Width", "Grain direction"])
    stockDF.loc[0, "@"] = "S"
    stockDF["Length"] = filteredStock["Height_in"].values[0] #.astype(str)
    stockDF["Width"] = filteredStock["Width_in"].values[0] #.astype(str)
    stockDF["Grain direction"] = ""
    
    # print(stockDF)

    return stockDF


def createHeaders(projectName, material, kref, partitions):
    # CREATING TITLE RAW MATERIAL NUMBER AND KREF VALUES AS HEADERS
    headDF = pd.DataFrame(["Code", "Value"])
    headDF.loc[0, "Code"] = "N"
    headDF.loc[0, "Value"] = f"{projectName} - X of {partitions}"

    headDF.loc[1, "Code"] = "M"
    headDF.loc[1, "Value"] = material

    headDF.loc[2, "Code"] = "K"
    headDF.loc[2, "Value"] = kref
    
    headDF.drop(columns=[0], inplace=True)
    # print(headDF)

    return headDF


def open_file_dialog():
    # Open a file dialog to select an Excel file
    
    file_path = filedialog.askopenfilename(
        title="Select an Excel File",
        filetypes=[("Excel Files", "*.xlsx;*.xls"), ("All Files", "*.*")]
    )
    
    if file_path:
        try:
            stockDF = prepare_stock_data(insDF, insDropdown.get())

            df = pd.read_excel(file_path)

            partList, rawMaterialList, uniqueParts = matchPartsAndRawMaterials(df['PART NUMBER'].tolist(), df['ITEM NO'].tolist(), df['QTY.'].tolist())

            df.dropna(subset=['ITEM NO'], inplace=True)
            df.drop(columns=["S. NO.", "PART NUMBER"], inplace=True)
            df.reset_index(drop=True, inplace=True)

            df["PART"] = partList
            df ["RAW MATERIAL"] = rawMaterialList

            cols = df.columns.tolist()
            cols = cols[-2:] + cols[:-2]
            df = df[cols]

            lables = []
            # while lables == [] or len(lables) != len(set(lables)) or "" in lables:
            lables = getLabels(uniqueParts)

            # print(df)

            # CREATE COMPLETE DTATFRAME WITH ALL REPEATED PARTS IN SEPARATE ROWS.
            # LETS SAY ONE INSULATION PART IS HAVING 2 QTY. ITS INDIVIDUAL CUTLIST ROWS ARE ADDED TWICE TO THE DATAFRAME.
            extendedDF = createExtendedDF(df, uniqueParts, lables)
            # print(extendedDF)

            extendedDF['ITEM NO'] = extendedDF['ITEM NO'].astype(int)

            extendedDF['LENGTH_FLOAT'] = extendedDF['LENGTH'].map(lambda x: x.rstrip('"'))
            extendedDF['LENGTH_FLOAT'] = extendedDF['LENGTH_FLOAT'].map(lambda x: float(sum(Fraction(term) for term in x.split())))
            extendedDF["LENGTH_FLOAT"] = extendedDF["LENGTH_FLOAT"].astype(float)

            extendedDF['WIDTH_FLOAT'] = extendedDF['WIDTH'].map(lambda x: x.rstrip('"'))
            extendedDF['WIDTH_FLOAT'] = extendedDF['WIDTH_FLOAT'].map(lambda x: float(sum(Fraction(term) for term in x.split())))
            extendedDF["WIDTH_FLOAT"] = extendedDF["WIDTH_FLOAT"].astype(float)

            # CALCULATE AREA OF THE PANELS WHICH IS USED TO GROUP THE PANELS LATER.
            # extendedDF['AREA'] = extendedDF['LENGTH_FLOAT']*extendedDF['WIDTH_FLOAT']
            # extendedDF['PERIMETER'] = 2*(extendedDF['LENGTH_FLOAT']+extendedDF['WIDTH_FLOAT'])

            # GROUPING PANELS BASED ON AREAS AND PERIMETER IS NOT WORKING IN CERTAIN CASES.
            # SO CALCULATING MINIMUM AND MAXIMUM DIMENSION OF THE PANELS WHICH IS USED TO GROUP THE PANELS LATER.
            extendedDF['MinDimension'] = extendedDF[['LENGTH_FLOAT', 'WIDTH_FLOAT']].min(axis=1)
            extendedDF['MaxDimension'] = extendedDF[['LENGTH_FLOAT', 'WIDTH_FLOAT']].max(axis=1)

            groupedDF = pd.DataFrame()
            
            groupedDF['PART'] = extendedDF.groupby(['MinDimension', 'MaxDimension'], group_keys=False)['PART'].apply(lambda x: ', '.join(x.astype(str))).reset_index()['PART']
            
            groupedDF['QTY. TRACK'] = extendedDF.groupby(['MinDimension', 'MaxDimension'], group_keys=False)['QTY.'].apply(lambda x: ', '.join(x.astype(str))).reset_index()['QTY.']
            groupedDF['ITEM TRACK'] = extendedDF.groupby(['MinDimension', 'MaxDimension'], group_keys=False)['ITEM NO'].apply(lambda x: ', '.join(x.astype(str))).reset_index()['ITEM NO']

            groupedDF['Quantity'] = extendedDF.groupby(['MinDimension', 'MaxDimension'], group_keys=False)['QTY.'].sum().reset_index()['QTY.']
            groupedDF['Grain direction'] = ""
            groupedDF['LABEL'] = extendedDF.groupby(['MinDimension', 'MaxDimension'], group_keys=False)['LABEL'].apply(lambda x: ', '.join(x.astype(str))).reset_index()['LABEL']

            groupedDF['Label'] = groupedDF.apply(lambda row: create_part_label(row['LABEL'], row['QTY. TRACK'], row['ITEM TRACK']), axis=1)
            groupedDF.drop(columns=['PART', 'QTY. TRACK', 'LABEL', "ITEM TRACK"], inplace=True)

            groupedDF['@'] = "P"
            groupedDF['Length'] = extendedDF.groupby(['MinDimension', 'MaxDimension'], group_keys=False)['LENGTH'].apply(lambda x: ', '.join(x.astype(str))).reset_index()['LENGTH']
            groupedDF['Length'] = groupedDF.apply(lambda row: remove_duplicates(row['Length']), axis=1)
            groupedDF['Width'] = extendedDF.groupby(['MinDimension', 'MaxDimension'], group_keys=False)['WIDTH'].apply(lambda x: ', '.join(x.astype(str))).reset_index()['WIDTH']
            groupedDF['Width'] = groupedDF.apply(lambda row: remove_duplicates(row['Width']), axis=1)

            cols = groupedDF.columns.tolist()
            cols = cols[-3:] + cols[:-3]
            groupedDF = groupedDF[cols]

            # print(groupedDF)
            # print(groupedDF[groupedDF["Label"].str.len() > 30])

            # processed_df = split_labels(groupedDF)
            groupedDF = split_labels(groupedDF)
            print(groupedDF)

            # print(df)  # Print the DataFrame to the console
            # print(uniqueParts)
            # print(lables)
            # print(extendedDF)

            n = 20  #chunk row size
            list_df = [groupedDF[i:i+n] for i in range(0,groupedDF.shape[0],n)]
            # print(list_df)

            headDF = createHeaders(project_name_entry.get(), insDropdown.get(), '3/16"', len(list_df))

            saveLocation = "/".join(file_path.split("/")[0:-1])
            fileName = f"{project_name_entry.get()}_{dropdown.get()}_OpticutterParts_({datetime.today().strftime('%Y%m%d')})"

            for i in range(len(list_df)):
                headDF.loc[0, "Value"] = f"{project_name_entry.get()} - {i+1} of {len(list_df)}"
                headDF.to_csv(f"{saveLocation}/{fileName}-{i+1}.csv", sep=',', encoding='utf-8', index=False, header=False)
                stockDF.to_csv(f"{saveLocation}/{fileName}-{i+1}.csv", sep=',', encoding='utf-8', index=False, header=True, mode='a')
                list_df[i].to_csv(f"{saveLocation}/{fileName}-{i+1}.csv", sep=',', encoding='utf-8', index=False, header=True, mode='a')
            
            # groupedDF.to_csv(f"{project_name_entry.get()}_{dropdown.get()}_OpticutterParts_({datetime.today().strftime('%Y%m%d')}).csv", sep=',', encoding='utf-8', index=False, header=True)

            tk.messagebox.showinfo("Success", "Successfully generated files. Exiting Script.")

            root.destroy()

        except Exception as e:
            print(f"Error loading Excel file: {e}")
            tk.messagebox.showerror("Error", "Unable to process the file. Exiting Script.")
            exit()


def float_to_fractional_inch(value, denominator=8):
    """Convert a float to a fractional inch string."""
    whole, decimal = divmod(value, 1)
    numerator = int(round(decimal * denominator))
    fraction = Fraction(numerator, denominator)
    if whole:
        if fraction.numerator == 0:
            return f"{int(whole)}\""
        return f"{int(whole)} {fraction}\""
    else:
        return f"{fraction}\""

# Create the main Tkinter window
root = tk.Tk()
root.title("Opticutter Input Generator")
root.geometry("400x400")

root.configure(bg=bg_color)

# Create the main heading label
heading_label = tk.Label(
    root,
    text="Opticutter Input Generator",
    font=("Helvetica", 16, "bold"),
    bg=bg_color,
    fg=fg_color
)
heading_label.pack(pady=10)

# Create the subheading label
subheading_label = tk.Label(
    root,
    text="Create OptiCutter parts input file\nusing grouped Cutlist BOM.",
    font=("Helvetica", 10),
    bg=bg_color,
    fg="#7a7a7a"  # Gray subheading text color
)
subheading_label.pack(pady=5)

# Create the project name label
project_name_label = tk.Label(
    root,
    text="Enter Project Name",
    bg=bg_color,
    fg=fg_color,
    anchor="w"
)
project_name_label.pack(pady=5, padx=20, fill='x')

# Create the project name entry
project_name_entry = tk.Entry(
    root,
    bg=entry_bg_color,
    fg=fg_color,
    insertbackground=fg_color
)
project_name_entry.insert(0, "")
project_name_entry.pack(pady=5, padx=20, ipadx=80, ipady=5, fill='x')

# Create the type label
type_label = tk.Label(
    root,
    text="Type",
    bg=bg_color,
    fg=fg_color,
    anchor="w"
)
type_label.pack(pady=5, padx=20, fill='x')

# Create a dropdown menu
options = ["Insulation"]     #["Insulation", "Floor Plate"]
selected_option = tk.StringVar()
selected_option.set(options[0])

dropdown = ttk.Combobox(
    root,
    textvariable=selected_option,
    values=options,
    state="readonly"
)
dropdown.pack(pady=5, padx=20, ipady=5, fill='x')

# Create the Raw-material label
rm_label = tk.Label(
    root,
    text="Raw-Material",
    bg=bg_color,
    fg=fg_color,
    anchor="w"
)
rm_label.pack(pady=5, padx=20, fill='x')

# Create a dropdown menu to select specific insulation
insDF = pd.read_excel("optiCutter_Script_Stocklist.xlsx", sheet_name="INSULATION")
insDF['Width_in'] = insDF['Width_in'].apply(float_to_fractional_inch)
insDF['Height_in'] = insDF['Height_in'].apply(float_to_fractional_inch)
insDF["OPTIONS"] = insDF["Name"] + " (" + insDF["Raw-Material"] + ")"

insOptions = insDF["OPTIONS"].tolist()
selectedIns = tk.StringVar()
selectedIns.set(insOptions[0])
insDropdown = ttk.Combobox(
    root,
    textvariable=selectedIns,
    values=insOptions,
    state="readonly"
)
insDropdown.pack(pady=5, padx=20, ipady=5, fill='x')

# Create the upload button
button = tk.Button(
    root,
    text="Upload BOM Spreadsheet",
    command=open_file_dialog,
    bg=button_color,
    fg=button_text_color,
    activebackground=button_text_color,
    activeforeground=button_color
)
button.pack(pady=15, padx=20, ipadx=10, ipady=5)

# Run the main loop
root.mainloop()

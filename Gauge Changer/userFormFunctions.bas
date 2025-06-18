Attribute VB_Name = "userFormFunctions"
Sub PopulateProfileList(partDict As Scripting.Dictionary)

    Dim key As Variant
    Dim PartProfileDict As New Scripting.Dictionary

    For Each key In partDict.Keys

         
        Dim Part As PartDoc
        Set Part = partDict(key)

        If Part.IsSheetMetal() Then
            
            If Not PartProfileDict.Exists(Part.Profile) Then
                
                PartProfileDict.Add Part.Profile, Part
    
                With UpdateGaugeForm
                
                    .availableProfilesComboBox.AddItem Part.Profile
                
                End With
                
                
            End If
'
'        Else
'
'            PartDict.Remove key
'
        End If
        
    Next
    
End Sub

Attribute VB_Name = "userFormFunctions"
Sub PopulateProfileList(partDict As Scripting.Dictionary)

    Dim key As Variant
    Dim PartProfileDict As New Scripting.Dictionary

    For Each key In partDict.Keys

         
        Dim Part As PartDoc
        Set Part = partDict(key)

        If Part.IsSheetMetal() Then
            
            If Not PartProfileDict.Exists(Part.Profile) And Not Part.Profile = "" Then
                
                PartProfileDict.Add Part.Profile, Part
    
                With UpdateGaugeForm
                
                    .availableProfilesComboBox.AddItem Part.Profile
                
                End With
                
                
            End If

        End If
        
    Next
    
End Sub

Function AddtoListBox(FormListBox As msforms.ListBox, FirstCol As String, SecondCol As String, ThirdCol As String)

    FormListBox.AddItem
    FormListBox.List(FormListBox.ListCount - 1, 0) = FirstCol
    FormListBox.List(FormListBox.ListCount - 1, 1) = SecondCol

    FormListBox.List(FormListBox.ListCount - 1, 2) = ThirdCol
    
End Function

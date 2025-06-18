VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "Append_Text"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Option Explicit
'Const VAULT_NAME As String = "FBD"

Public Sub Append_text_file(Macro As String)

'    Dim pdmVault As EdmVault5
'    Set pdmVault = New EdmVault5
'
'    pdmVault.LoginAuto VAULT_NAME, 0
'
'    If pdmVault.IsLoggedIn Then
'
'        Dim userMgr As IEdmUserMgr5
'        Set userMgr = pdmVault
'
'        Dim user As IEdmUser5
'        Set user = userMgr.GetLoggedInUser
    
        Dim strFile_Path As String
        strFile_Path = DropboxPath() & "\COMMON\Useful Information\MacroLogs.txt"
'        Debug.Print strFile_Path
On Error GoTo Error:
        Open strFile_Path For Append As #1
             Write #1, Macro & "? RUN BY    ?" & Environ$("username") & " ?   ON    ?" & Now()
        Close #1
'    End If
Error:
End Sub

Private Function DropboxPath() As String

   Dim RegEx As Object
   Dim MatchColl As Object
   Dim DataLine As String
   Const FileNum = 1 ' Assumes no other files are open!!
   
   Set RegEx = CreateObject("VBScript.RegExp")
   RegEx.Global = True
   RegEx.IgnoreCase = False
On Error GoTo Error2:
   Open Environ("LOCALAPPDATA") & "\Dropbox\info.json" For Input As #FileNum
'    Do While Not EOF(FileNum)
     DataLine = Input(LOF(FileNum), FileNum)
'    Loop
   Close #FileNum

   RegEx.Pattern = "^.*""path"": ""([^""]*).*"

   DropboxPath = Replace(RegEx.Replace(DataLine, "$1"), "", "")
    
   DropboxPath = Replace(DropboxPath, "\\", "\")
'   Debug.Print DropboxPath
Error2:
'    MsgBox "dpath not worked"
    
End Function


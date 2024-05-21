import QtQuick
import QtQuick.Controls
import QtQuick.Window
Window {
    id: main_window
    property string imagePath: "images/confused.png"
    
    width: 1920
    height: 1080
    visible: true
    title: qsTr("Hello World")
    Connections{
        target: test_back_end
    }
    Image {
        id: image
        anchors.horizontalCenter: parent.horizontalCenter
        width: 739
        height: 602
        source: "images/logo.png"
        fillMode: Image.PreserveAspectFit
    }
    Text{
        id: test
        text: "ahhaha"
        MouseArea{
            anchors.fill: parent
            onClicked: test.text = "dsadsa"
        }
    }
    Rectangle {
        id: buttons_container
        color: "white"
        width: 539
        height: 46
        border.width: 1
        anchors {
            horizontalCenter: parent.horizontalCenter
            verticalCenter: parent.verticalCenter
        }
        
        property int activeButton: 0
        
        Rectangle {
            id: naive_bayes_button
            width: 184
            height: 46
            anchors.right: parent.right
            border.width: buttons_container.activeButton === 1 ? 2 : 0
            color: buttons_container.activeButton === 1 ? "#4EA0CF" : "transparent"
            
            Text {
                text: "NB"
                font.bold: true
                font.pixelSize: 16
                anchors {
                    horizontalCenter: parent.horizontalCenter
                    verticalCenter: parent.verticalCenter
                }
            }
            
            MouseArea {
                id: naive_bayes_button_mouse
                anchors.fill: parent
                hoverEnabled: true
                onClicked: buttons_container.activeButton = 1
            }
        }
        
        Rectangle {
            id: logistic_reg_button
            width: 184
            height: 46
            anchors.right: naive_bayes_button.left
            color: buttons_container.activeButton === 2 ? "#92BF30" : "transparent"
            border.width: buttons_container.activeButton === 2 ? 2 : 0
            
            Text {
                text: "LR"
                font.bold: true
                font.pixelSize: 16
                anchors {
                    horizontalCenter: parent.horizontalCenter
                    verticalCenter: parent.verticalCenter
                }
            }
            
            MouseArea {
                id: logistic_reg_button_mouse
                anchors.fill: parent
                hoverEnabled: true
                onClicked: buttons_container.activeButton = 2
            }
        }
        
        Rectangle {
            id: svm_button
            width: 184
            height: 46
            anchors.right: logistic_reg_button.left
            color: buttons_container.activeButton === 3 ? "#FA8830" : "transparent"
            border.width: buttons_container.activeButton === 3 ? 2 : 0
            
            Text {
                text: "SVM"
                font.bold: true
                font.pixelSize: 16
                anchors {
                    horizontalCenter: parent.horizontalCenter
                    verticalCenter: parent.verticalCenter
                }
            }
            
            MouseArea {
                id: svm_button_mouse
                anchors.fill: parent
                hoverEnabled: true
                onClicked: buttons_container.activeButton = 3
            }
        }
    }
    
    Rectangle {
        width: 1078
        height: 120
        border.width: 2
        radius: 15
        anchors {
            horizontalCenter: parent.horizontalCenter
            bottom: parent.bottom
            bottomMargin: 45
        }
        
        TextInput {
            id: text_area
            width: parent.width * 0.75
            height: parent.height * 0.75
            text: "type here..."
            clip: true
            font.bold: true
            font.pointSize: 24
            anchors {
                top: parent.top
                left: parent.left
                topMargin: 15
                leftMargin: 15
            }
        }
        Rectangle{
            id: myCircleButton
            color: submitButton.pressed ? "silver" : "white"
            border.width: 3
            border.color: "black"
            width: 100
            height: 100
            radius: width / 2
            anchors{
                right: text_area.left
                rightMargin: 40
            }
            Text{
                text: "submit"
                font.pixelSize: 20
            anchors{
                horizontalCenter: parent.horizontalCenter
                verticalCenter: parent.verticalCenter
            }
            }
            MouseArea{
                id: submitButton
                anchors.fill: parent
                onClicked:{
                    test.text = test_back_end.predict(buttons_container.activeButton, text_area.text)
                    if(test.text === "Positive")
                        main_window.imagePath = "images/positive.png"
                    else if(test.text === "Negative")
                        main_window.imagePath = "images/negative.png"
                    else if(test.text === "Neutral")
                        main_window.imagePath = "images/neutral.png"
                    else
                        main_window.imagePath = "images/confused.png"
                }
            }
        }
    }
    Image{
        id: emoji
        width: 215
        height: 215
        source: main_window.imagePath
        y: 600
        anchors{
            horizontalCenter: parent.horizontalCenter
            // verticalCenter: parent.verticalCenter
        }
        fillMode: Image.PreserveAspectFit
    }
}

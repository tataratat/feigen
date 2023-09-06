import feigen

if __name__ == "__main__":
    # feigen.log.configure(debug=True)

    b = feigen.BSpline2D("ws://localhost:9001")
    b.start()

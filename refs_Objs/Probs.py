import pandas


def open_data():
    col_names = ['Ranking', 'Team', 'Points', 'P', 'W', 'D', 'L', 'GS', 'GR']
    data = pandas.read_csv('E:\Hackathon\SPL.csv', names=col_names, converters={"W": int, "P": int})
    print(data.W.describe(exclude='object'))

    total_number_of_teams = sum(1 for row in data.Ranking.tolist())
    t_no_matches_for_each = (total_number_of_teams-1) * 2

    winning_matches = data.W.tolist()
    played_matches = data.P.tolist()
    draw_matches = data.D.tolist()
    lose_matches = data.L.tolist()
    prob_for_win_matches = []
    prob_for_draw_matches = []
    prob_for_lose_matches = []

    for iterate in range(len(winning_matches)):
        prob_for_win_matches.append((winning_matches[iterate]*t_no_matches_for_each)/played_matches[iterate])
        prob_for_draw_matches.append((draw_matches[iterate]*t_no_matches_for_each)/played_matches[iterate])
        prob_for_lose_matches.append((lose_matches[iterate]*t_no_matches_for_each)/played_matches[iterate])

    print("Total Number of Matches", total_number_of_teams)
    print("Every Team will play", t_no_matches_for_each)
    print("Total Winning Matches", winning_matches)
    print("Total Played Matches", played_matches)
    print("Probabilities for team winning: ", prob_for_win_matches)
    print("Probabilities for team Drawing: ", prob_for_draw_matches)
    print("Probabilities for team losing: ", prob_for_lose_matches)


open_data()




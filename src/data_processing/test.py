if __name__ == '__main__':
    list_of_relationships = [["1", "2", "3"], ["1", "2", "3"], ["2", "3", "2"]]

    list_of_relationships_tmp = set(tuple(relationship) for relationship in list_of_relationships)

    # every edge from tuple to list
    list_of_relationships = list(list(relationship) for relationship in list_of_relationships_tmp)

    print(list_of_relationships)
